"""
Dataset over the cached proposals, with canonicalisation applied in the
collate function.

Canonicalisation happens per *batch*, not per sample, so it runs through the
numba kernels in bench_canon.py. Measured on this workload: batched analytic
PCA is 0.098 ms for 64 clusters, versus 1.50 ms for np.linalg.eigh in a Python
loop -- 15x, and the loop version would sit in the DataLoader worker and starve
the GPU.

Channels handed to the network (in_ch = 6):

    0-2  canonicalised xyz, unit sphere
    3    height above ground     <- invariant to yaw, and to whatever the
    4    range from sensor       <- canonicaliser gets wrong
    5    intensity

Channels 3-5 are the insurance policy. If the canonical frame is inconsistent
across viewpoints -- which canon_study.py shows it is -- these still carry
usable signal, because none of them changes when the object rotates about z.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .bench_canon import pca2_batch, pca3_batch
from .boxes import encode_heading
from .config import Config

# KITTI mean dimensions (l, w, h). Size is regressed as a log-ratio against
# these so a 0.8 m pedestrian and a 3.9 m car produce comparably scaled targets.
ANCHORS = np.array([
    [1.00, 1.00, 1.00],     # background - unused
    [3.89, 1.62, 1.53],     # Car
    [0.84, 0.66, 1.76],     # Pedestrian
    [1.76, 0.60, 1.74],     # Cyclist
], dtype=np.float32)


class ProposalSet(Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        shards = sorted(cfg.cache_dir.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(
                f"no shards in {cfg.cache_dir}. Run: python -m pnd.proposals")

        pts, meta, frame = [], [], []
        for s in shards:
            z = np.load(s)
            pts.append(z["points"])
            meta.append(z["meta"])
            frame.append(z["frame"])
        self.points = np.concatenate(pts)
        self.meta = np.concatenate(meta)
        self.frame = np.concatenate(frame)

        # split by FRAME, never by proposal: two clusters from the same scan
        # share ground plane, weather and pose, so splitting by proposal leaks
        # the validation set into training and flatters the numbers.
        uf = np.unique(self.frame)
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(uf)
        n_val = max(int(len(uf) * cfg.val_frac), 1)
        val_frames = set(uf[:n_val].tolist())
        in_val = np.array([f in val_frames for f in self.frame])
        keep = in_val if split == "val" else ~in_val

        self.points = self.points[keep]
        self.meta = self.meta[keep]
        self.frame = self.frame[keep]
        self.split = split

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int):
        return self.points[i], self.meta[i]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.meta[:, 0].astype(int),
                           minlength=self.cfg.num_classes)


# --------------------------------------------------------------------------- #
def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3), np.float64)
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


class Collate:
    """Batched canonicalisation + feature assembly."""

    def __init__(self, cfg: Config, train: bool):
        self.cfg = cfg
        self.train = train

    def __call__(self, batch):
        pts = np.stack([b[0] for b in batch]).astype(np.float64)   # (B, P, 5)
        meta = np.stack([b[1] for b in batch]).astype(np.float64)  # (B, 9)
        B, P, _ = pts.shape
        cfg = self.cfg

        xyz = pts[:, :, :3].copy()
        inten = pts[:, :, 3]
        agl = pts[:, :, 4]
        rng_raw = np.linalg.norm(xyz, axis=2)          # before any rotation

        ctr_gt = meta[:, 1:4].copy()
        dims_gt = meta[:, 4:7].copy()
        yaw_gt = meta[:, 7].copy()
        cls = meta[:, 0].astype(np.int64)

        # ---- augmentation ---------------------------------------------- #
        if self.train:
            # yaw: rotate about the sensor z axis. Free, exactly matches the
            # real nuisance, and unlike canonicalisation it cannot be
            # inconsistent across viewpoints.
            if cfg.yaw_aug:
                a = np.random.uniform(-np.pi, np.pi, B)
                R = _rot_z(a)
                xyz = np.einsum("bij,bpj->bpi", R, xyz)
                ctr_gt = np.einsum("bij,bj->bi", R, ctr_gt)
                yaw_gt = yaw_gt + a

            # mirror about x. A car seen from the left is a valid car seen from
            # the right; heading negates with it.
            if cfg.aug_flip:
                f = np.random.rand(B) < 0.5
                xyz[f, :, 1] *= -1.0
                ctr_gt[f, 1] *= -1.0
                yaw_gt[f] *= -1.0

            # point dropout: keep a random fraction, then refill to P by
            # repeating what survived. Simulates the same object arriving with
            # far fewer returns because it is further away.
            if cfg.aug_dropout > 0:
                for b in range(B):
                    keep = np.random.uniform(1.0 - cfg.aug_dropout, 1.0)
                    k = max(int(P * keep), 8)
                    idx = np.random.choice(P, k, replace=False)
                    fill = np.random.choice(idx, P, replace=True)
                    xyz[b] = xyz[b][fill]
                    inten[b] = inten[b][fill]
                    agl[b] = agl[b][fill]
                    rng_raw[b] = rng_raw[b][fill]

            # scale: object-size variation the anchors do not cover.
            # Scale about the cluster centroid, NOT the sensor origin. Scaling
            # raw coordinates moves an object at 20 m by 1.6 m at 8% -- a large
            # translation masquerading as a size change, which corrupts the
            # centre target and desynchronises the range/height channels.
            if cfg.aug_scale > 0:
                sc = np.random.uniform(1 - cfg.aug_scale, 1 + cfg.aug_scale, B)
                c0 = xyz.mean(axis=1, keepdims=True)          # (B, 1, 3)
                xyz = c0 + (xyz - c0) * sc[:, None, None]
                ctr_gt = c0[:, 0, :] + (ctr_gt - c0[:, 0, :]) * sc[:, None]
                dims_gt = dims_gt * sc[:, None]

            # jitter: sensor range noise, a couple of centimetres
            if cfg.aug_jitter > 0:
                xyz += np.random.normal(0.0, cfg.aug_jitter, xyz.shape)

        # ---- canonicalise --------------------------------------------- #
        flat = np.ascontiguousarray(xyz.reshape(-1, 3))
        offs = (np.arange(B + 1) * P).astype(np.int64)
        Rc = np.zeros((B, 3, 3)); tc = np.zeros((B, 3)); lam = np.zeros((B, 3))

        mode = cfg.canon
        if mode in ("none", "tnet3"):
            tc = xyz.mean(axis=1)
            Rc = np.repeat(np.eye(3)[None], B, 0)
        elif mode == "pca3_skew":
            pca3_batch(flat, offs, Rc, tc, lam)
        elif mode in ("pca2_yaw", "pca4_ensemble"):
            yaw_c = np.zeros(B)
            pca2_batch(flat, offs, yaw_c)
            tc = xyz.mean(axis=1)
            Rc = _rot_z(-yaw_c)
        else:
            raise ValueError(mode)

        cen = xyz - tc[:, None, :]
        xc = np.einsum("bij,bpj->bpi", Rc, cen)
        scale = np.maximum(np.linalg.norm(xc, axis=2).max(axis=1), 1e-6)
        xc = xc / scale[:, None, None]

        # ---- targets, in the same canonical frame --------------------- #
        dc = np.einsum("bij,bj->bi", Rc, ctr_gt - tc) / scale[:, None]
        anch = ANCHORS[np.clip(cls, 0, 3)]
        size_log = np.log(np.maximum(dims_gt, 1e-3) / np.maximum(anch, 1e-3))
        yaw_c_off = np.arctan2(Rc[:, 1, 0], Rc[:, 0, 0])
        yaw_t = yaw_gt + yaw_c_off

        feats = np.concatenate([
            xc,
            (agl / 3.0)[:, :, None],
            (rng_raw / cfg.max_range)[:, :, None],
            inten[:, :, None],
        ], axis=2).transpose(0, 2, 1)                  # (B, 6, P)

        hb, hr = encode_heading(yaw_t)
        out = {
            "x": torch.from_numpy(feats).float(),
            "cls": torch.from_numpy(cls),
            "center": torch.from_numpy(dc).float(),
            "size_log": torch.from_numpy(size_log).float(),
            "head_bin": torch.from_numpy(hb),
            "head_res": torch.from_numpy(hr).float(),
            "dims": torch.from_numpy(dims_gt).float(),
            "anchor": torch.from_numpy(anch.astype(np.float64)).float(),
            "scale": torch.from_numpy(scale).float(),
        }

        if mode == "pca4_ensemble":
            # the four det=+1 sign combinations of the two footprint axes.
            # PCAlign's argument: do not choose a sign, enumerate and let the
            # network pick the frame it is most confident in.
            copies = []
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                f = out["x"].clone()
                f[:, 0] *= sx
                f[:, 1] *= sy
                copies.append(f)
            out["x"] = torch.stack(copies, dim=1)      # (B, 4, 6, P)
        return out


def loaders(cfg: Config):
    tr = ProposalSet(cfg, "train")
    va = ProposalSet(cfg, "val")
    from torch.utils.data import DataLoader
    mk = lambda ds, trn: DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=trn, drop_last=trn,
        num_workers=cfg.num_workers, collate_fn=Collate(cfg, trn),
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=cfg.num_workers > 0)
    return mk(tr, True), mk(va, False), tr
