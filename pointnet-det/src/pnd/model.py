"""
PointNet backbone and heads for cluster-wise 3D detection.

DESIGN DECISIONS AND THE EVIDENCE FOR THEM
------------------------------------------

1. The 64x64 feature T-Net is deleted, not replaced.

   From the original PointNet ablation on ModelNet40:

       no transform                     87.1
       input  (3x3)                     87.9   +0.8
       feature (64x64)                  86.9   -0.2   <- worse than nothing
       feature (64x64) + reg            87.4   +0.3
       both + reg                       89.2   +2.1

   The feature transform *alone hurts*. It only pays off jointly with the input
   transform and an orthogonality regulariser, and it is by far the most
   expensive piece (it regresses 4096 numbers from a full mini-PointNet). PCA
   cannot substitute for it either -- PCA is a geometric operation and a learned
   64-d feature space has no geometry. So it is removed outright.

2. The 3x3 input transform is a config choice, not a fixture.

   `canon=tnet3` keeps PointNet's learned version. The `pca*` modes replace it
   with a deterministic frame computed on CPU. `canon=none` is the control.
   See canon.py for what each does and canon_study.py for how they behave under
   partial observation.

3. Extra input channels are rotation-invariant on purpose.

   Height above ground, range, and intensity do not change when the object
   rotates about z, so they give the network signal that survives whatever the
   canonicaliser does or fails to do. This is cheap insurance against a bad
   frame.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boxes import NUM_HEADING_BINS


# --------------------------------------------------------------------------- #
def _mlp1d(dims, bn=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class TNet3(nn.Module):
    """PointNet's learned 3x3 input transform.

    Kept deliberately narrower than the paper's (64/128/512 rather than
    64/128/1024). The 1024-wide activation is the memory-bandwidth bottleneck --
    measured at 554 us per cluster on a GTX 1650 -- and a 3x3 output does not
    need that much capacity. This is the fair version of the baseline: if we are
    going to argue T-Net is expensive, we should not inflate it first.
    """

    def __init__(self, width: int = 512):
        super().__init__()
        self.conv = _mlp1d([3, 64, 128, width])
        self.fc = nn.Sequential(
            nn.Linear(width, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 9),
        )
        # start at identity: last layer zeroed, identity added in forward
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:   # (B, 3, P)
        f = self.conv(xyz).max(dim=2).values
        m = self.fc(f).view(-1, 3, 3)
        eye = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
        return m + eye


# --------------------------------------------------------------------------- #
class PointNetEncoder(nn.Module):
    """Shared per-point MLP followed by a symmetric max over points.

    The max is what makes this permutation invariant, which is the whole reason
    the architecture works on a set. Nothing here may depend on point order.
    """

    def __init__(self, in_ch: int = 3, width: int = 1024, use_tnet3: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.tnet3 = TNet3() if use_tnet3 else None
        self.mlp1 = _mlp1d([in_ch, 64, 64])
        self.mlp2 = _mlp1d([64, 128, width])
        self.width = width

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """x: (B, C, P) with the first three channels xyz.

        Returns (global feature, per-point features, tnet matrix or None).
        """
        m = None
        if self.tnet3 is not None:
            xyz = x[:, :3, :]
            m = self.tnet3(xyz)
            x = torch.cat([torch.bmm(m, xyz), x[:, 3:, :]], dim=1) \
                if x.shape[1] > 3 else torch.bmm(m, xyz)
        local = self.mlp1(x)
        g = self.mlp2(local).max(dim=2).values
        return g, local, m


class ClusterNet(nn.Module):
    """Classify one cluster and regress its 3D box.

    Boxes are predicted in the *canonical* frame; canon.Frame inverts them back
    to velodyne. Size is predicted as a log-ratio against a per-class anchor,
    which keeps the regression well scaled across a 0.5 m pedestrian and a 4 m
    car. Yaw is predicted as (sin, cos) because a raw angle is discontinuous at
    +/-pi and the network cannot represent that jump.
    """

    def __init__(self, num_classes: int = 4, in_ch: int = 6,
                 width: int = 1024, use_tnet3: bool = False,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(in_ch, width, use_tnet3)
        self.trunk = nn.Sequential(
            nn.Linear(width, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(256, num_classes)
        # centre(3) + log size ratio(3) + heading bin logits + bin residuals
        self.box_head = nn.Linear(256, 3 + 3 + 2 * NUM_HEADING_BINS)

    def forward(self, x: torch.Tensor) -> dict:
        g, _, m = self.encoder(x)
        h = self.trunk(g)
        box = self.box_head(h)
        nb = NUM_HEADING_BINS
        return {
            "logits": self.cls_head(h),
            "center": box[:, 0:3],
            "size_log": box[:, 3:6],
            "head_bin": box[:, 6:6 + nb],
            "head_res": box[:, 6 + nb:6 + 2 * nb],
            "tnet": m,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
class MultiFramePointNet(nn.Module):
    """PCAlign-style: run the shared encoder over K sign-consistent PCA frames.

    Rather than *choosing* an eigenvector sign -- which skewness does, but which
    is only well posed when the distribution is genuinely asymmetric -- this
    enumerates the four det=+1 sign combinations and lets the network pick.

    Merging is a max over frames of the per-class logit, following PCAlign's
    "maximum probability selection": summing would let one badly-posed frame
    drag down three good ones.

    Cost is K forward passes of the encoder. Cheap here only because the encoder
    is small and clusters are small.
    """

    def __init__(self, num_classes: int = 4, in_ch: int = 6,
                 width: int = 1024, n_frames: int = 4, dropout: float = 0.3):
        super().__init__()
        self.n_frames = n_frames
        self.net = ClusterNet(num_classes, in_ch, width,
                              use_tnet3=False, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, K, C, P) -- K pre-canonicalised copies per cluster."""
        B, K = x.shape[0], x.shape[1]
        flat = x.reshape(B * K, x.shape[2], x.shape[3])
        out = self.net(flat)
        logits = out["logits"].view(B, K, -1)
        # pick the frame the network is most confident about, per sample
        conf = logits.softmax(-1).max(-1).values          # (B, K)
        pick = conf.argmax(dim=1)                          # (B,)
        idx = pick.view(B, 1, 1)
        sel = lambda t: t.view(B, K, -1).gather(
            1, idx.expand(-1, -1, t.shape[-1])).squeeze(1)
        return {
            "logits": sel(out["logits"]),
            "center": sel(out["center"]),
            "size_log": sel(out["size_log"]),
            "head_bin": sel(out["head_bin"]),
            "head_res": sel(out["head_res"]),
            "frame_idx": pick,
            "tnet": None,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(cfg) -> nn.Module:
    canon = getattr(cfg, "canon", "none")
    if canon == "pca4_ensemble":
        return MultiFramePointNet(cfg.num_classes, cfg.in_ch, cfg.width,
                                  n_frames=4, dropout=cfg.dropout)
    return ClusterNet(cfg.num_classes, cfg.in_ch, cfg.width,
                      use_tnet3=(canon == "tnet3"), dropout=cfg.dropout)
