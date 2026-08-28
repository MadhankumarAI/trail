"""
Box parameterisation: corner geometry and binned heading.

Two changes from the first version, both driven by what the IoU sweep measured.

CORNER LOSS instead of independent smooth-L1 on centre and size
---------------------------------------------------------------
The model was trained on smooth-L1 over (centre, log-size, sin/cos yaw) but
scored on IoU. Those are not the same objective. Smooth-L1 treats a 0.33 m
centre error identically on a 3.89 m car and a 0.66 m-wide pedestrian, but IoU
does not: measured AP at moderate difficulty fell 57.4 -> 26.2 for Car between
IoU 0.3 and 0.7, and 57.2 -> 1.3 for Pedestrian.

Corner loss (Frustum PointNets, PointRCNN) puts an L1 on the eight box corners
in metric space, so centre, size and heading errors are penalised through their
combined geometric effect -- which is what IoU actually measures. Dividing by
the class anchor diagonal makes it scale-relative, so an error that would ruin a
pedestrian's IoU costs proportionally more than the same error on a car.

BINNED HEADING instead of direct (sin, cos) regression
------------------------------------------------------
A car viewed from behind looks almost exactly like one viewed from the front, so
the heading posterior is bimodal at yaw and yaw+pi. Regressing a single angle
makes the network hedge between the two modes and land halfway between them --
measured heading error stuck at 17.4 degrees, which is expensive at IoU 0.7 on a
4 m box. Classifying a coarse bin and regressing a residual inside it lets the
network commit to a mode. This is what SECOND, PointPillars and CenterPoint all
do.
"""
from __future__ import annotations

import numpy as np
import torch

NUM_HEADING_BINS = 12                       # 30 degrees per bin
BIN_WIDTH = 2.0 * np.pi / NUM_HEADING_BINS


# --------------------------------------------------------------------------- #
# heading encode / decode
# --------------------------------------------------------------------------- #
def encode_heading(yaw: np.ndarray):
    """(N,) angle -> (bin index, residual normalised to [-0.5, 0.5])."""
    shifted = (yaw + BIN_WIDTH / 2.0) % (2.0 * np.pi)
    b = np.floor(shifted / BIN_WIDTH).astype(np.int64) % NUM_HEADING_BINS
    centre = b * BIN_WIDTH
    res = (yaw - centre + np.pi) % (2.0 * np.pi) - np.pi
    return b, res / BIN_WIDTH


def decode_heading(bin_idx, residual):
    """Inverse of encode_heading. Works for numpy or torch."""
    return bin_idx * BIN_WIDTH + residual * BIN_WIDTH


# --------------------------------------------------------------------------- #
# corners
# --------------------------------------------------------------------------- #
def corners_torch(center: torch.Tensor, dims: torch.Tensor,
                  yaw: torch.Tensor) -> torch.Tensor:
    """(B,3) centre, (B,3) l/w/h, (B,) yaw  ->  (B, 8, 3) corners.

    Corner order is fixed, so the L1 between two corner sets is only meaningful
    when both boxes use this same ordering -- which they do, because both go
    through this function.
    """
    l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    x = torch.cat([l, l, -l, -l, l, l, -l, -l], dim=1) / 2.0
    y = torch.cat([w, -w, -w, w, w, -w, -w, w], dim=1) / 2.0
    z = torch.cat([-h, -h, -h, -h, h, h, h, h], dim=1) / 2.0

    c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    xr = x * c - y * s
    yr = x * s + y * c
    return torch.stack([xr, yr, z], dim=2) + center.unsqueeze(1)


def corner_loss(pred_center, pred_dims, pred_yaw,
                gt_center, gt_dims, gt_yaw, anchor_diag,
                huber_beta: float = 1.0) -> torch.Tensor:
    """Scale-relative corner L1, invariant to a 180 degree heading flip.

    The flip term matters: a box rotated by pi occupies exactly the same volume,
    so penalising the network for choosing the opposite end of a symmetric
    object teaches it nothing and fights the binned-heading head.
    """
    pc = corners_torch(pred_center, pred_dims, pred_yaw)
    gc = corners_torch(gt_center, gt_dims, gt_yaw)
    gcf = corners_torch(gt_center, gt_dims, gt_yaw + np.pi)

    d = anchor_diag.clamp(min=1e-3).unsqueeze(1).unsqueeze(2)
    la = torch.nn.functional.smooth_l1_loss(pc / d, gc / d, beta=huber_beta,
                                            reduction="none").mean(dim=(1, 2))
    lf = torch.nn.functional.smooth_l1_loss(pc / d, gcf / d, beta=huber_beta,
                                            reduction="none").mean(dim=(1, 2))
    return torch.minimum(la, lf).mean()


def anchor_diagonal(anchors: torch.Tensor) -> torch.Tensor:
    """(B,3) l/w/h -> (B,) body diagonal, used to make the loss scale-relative."""
    return torch.linalg.norm(anchors, dim=1)
