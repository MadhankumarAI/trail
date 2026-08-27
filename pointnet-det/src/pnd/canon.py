"""
Deterministic canonicalisation - the replacement for PointNet's input T-Net.

Four variants, selected by config, all with the same signature so the ablation
is a one-line change:

    none        centre and scale only. No rotation. The control.
    tnet3       learned 3x3 (lives in model.py; this module returns identity)
    pca3_skew   full 3D PCA, eigenvector signs fixed by third moment
    pca2_yaw    2D PCA on (x, y) only - yaw canonicalised, gravity preserved

Every function returns (canonical_points, R, t, scale) so the transform can be
inverted: a box predicted in the canonical frame maps back to velodyne with

    center_velo = R.T @ (center_canon * scale) + t
    yaw_velo    = yaw_canon + yaw_of(R)

WHY pca3_skew IS SUSPECT
------------------------
Sign ambiguity (v vs -v) is the *easy* failure and third-moment sign-fixing does
handle it. Two harder ones remain, and both are measured here rather than
assumed:

1. Eigenvalue degeneracy. When two eigenvalues are close the eigenvectors are
   arbitrary - any rotation within that eigenspace is equally valid - so the
   frame jumps discontinuously between near-identical inputs. `degeneracy()`
   reports lambda2/lambda1 and lambda3/lambda2.

2. Partial observation. PCA of a *visible surface* is not PCA of the *object*.
   A car seen from behind has different principal axes than the same car seen
   from the side, so PCA does not canonicalise consistently across viewpoints -
   which is the entire point of canonicalising. Skewness is especially fragile
   here because the visible surface is asymmetric by construction: you only ever
   see the near side, so the third moment is dominated by sensor position rather
   than object shape.

`stability_report()` measures both directly on real clusters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

MODES = ("none", "tnet3", "pca3_skew", "pca2_yaw")

# Below this |skewness| the sign of an eigenvector is not determined by the
# data - the distribution is symmetric along that axis, so the third moment is
# noise. Flagged rather than silently accepted.
SKEW_EPS = 1e-3

# Two eigenvalues within this ratio of each other leave their eigenvectors
# effectively arbitrary.
DEGENERACY_RATIO = 0.85


@dataclass
class Frame:
    """The rigid transform applied, so predictions can be mapped back."""
    R: np.ndarray            # (3, 3) rotation, world -> canonical
    t: np.ndarray            # (3,) translation removed before rotating
    scale: float             # divisor applied after rotating
    degenerate: bool = False  # eigen-decomposition was ill-conditioned
    skew_ambiguous: bool = False  # at least one sign was undetermined

    def invert_points(self, pts_canon: np.ndarray) -> np.ndarray:
        return (self.R.T @ (pts_canon * self.scale).T).T + self.t

    def invert_yaw(self, yaw_canon: float) -> float:
        return yaw_canon + np.arctan2(self.R[1, 0], self.R[0, 0])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unit_scale(x: np.ndarray) -> float:
    """Scale to the unit sphere, as the original PointNet does."""
    s = float(np.max(np.linalg.norm(x, axis=1)))
    return s if s > 1e-6 else 1.0


def _right_handed(R: np.ndarray) -> np.ndarray:
    """Force det(R) = +1 so the frame is a rotation, not a reflection."""
    if np.linalg.det(R) < 0:
        R = R.copy()
        R[2] = -R[2]
    return R


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def canon_none(pts: np.ndarray) -> Tuple[np.ndarray, Frame]:
    """Centre and scale. No rotation at all - the control arm."""
    t = pts[:, :3].mean(axis=0)
    x = pts[:, :3] - t
    s = _unit_scale(x)
    return x / s, Frame(R=np.eye(3), t=t, scale=s)


def canon_pca3_skew(pts: np.ndarray) -> Tuple[np.ndarray, Frame]:
    """Full 3D PCA with third-moment sign disambiguation.

    Axes are ordered by descending eigenvalue, each sign chosen so the point
    distribution is positively skewed along it, then the frame is forced
    right-handed.
    """
    t = pts[:, :3].mean(axis=0)
    x = pts[:, :3] - t

    cov = (x.T @ x) / max(len(x) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    ambiguous = False
    axes = []
    for i in range(3):
        v = evecs[:, i]
        proj = x @ v
        # third moment; normalised so the threshold is scale free
        sd = proj.std()
        skew = float(np.mean(proj ** 3) / (sd ** 3 + 1e-12)) if sd > 1e-9 else 0.0
        if abs(skew) < SKEW_EPS:
            ambiguous = True                    # symmetric: sign is arbitrary
        if skew < 0:
            v = -v
        axes.append(v)

    R = _right_handed(np.stack(axes, axis=0))   # rows are the new basis
    lam = np.maximum(evals, 1e-12)
    degenerate = bool(lam[1] / lam[0] > DEGENERACY_RATIO or
                      lam[2] / lam[1] > DEGENERACY_RATIO)

    xc = (R @ x.T).T
    s = _unit_scale(xc)
    return xc / s, Frame(R=R, t=t, scale=s,
                         degenerate=degenerate, skew_ambiguous=ambiguous)


def canon_pca2_yaw(pts: np.ndarray) -> Tuple[np.ndarray, Frame]:
    """Yaw-only canonicalisation from 2D PCA of the (x, y) projection.

    Gravity is a real, meaningful direction in driving data - cars sit on roads,
    poles are vertical, pedestrians are upright. Rotating that away throws
    information out. The only genuine nuisance degree of freedom is heading, so
    only heading is removed.

    The 2D covariance is also far better conditioned than the 3D one: for almost
    any road object the footprint is clearly elongated, whereas the full 3D
    eigenvalues are frequently near-degenerate.
    """
    t = pts[:, :3].mean(axis=0)
    x = pts[:, :3] - t

    xy = x[:, :2]
    cov = (xy.T @ xy) / max(len(xy) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    v = evecs[:, 0]                             # dominant footprint direction
    proj = xy @ v
    sd = proj.std()
    skew = float(np.mean(proj ** 3) / (sd ** 3 + 1e-12)) if sd > 1e-9 else 0.0
    ambiguous = abs(skew) < SKEW_EPS
    if skew < 0:
        v = -v

    yaw = float(np.arctan2(v[1], v[0]))
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s_, 0.0],
                  [s_,  c, 0.0],
                  [0.0, 0.0, 1.0]])

    lam = np.maximum(evals, 1e-12)
    degenerate = bool(lam[1] / lam[0] > DEGENERACY_RATIO)

    xc = (R @ x.T).T
    s = _unit_scale(xc)
    return xc / s, Frame(R=R, t=t, scale=s,
                         degenerate=degenerate, skew_ambiguous=ambiguous)


def canonicalize(pts: np.ndarray, mode: str) -> Tuple[np.ndarray, Frame]:
    if mode in ("none", "tnet3"):
        # tnet3 does its rotation inside the network; here it is centre+scale
        return canon_none(pts)
    if mode == "pca3_skew":
        return canon_pca3_skew(pts)
    if mode == "pca2_yaw":
        return canon_pca2_yaw(pts)
    raise ValueError(f"unknown canon mode {mode!r}, expected one of {MODES}")


# --------------------------------------------------------------------------- #
# diagnostics - the experiment that settles the argument
# --------------------------------------------------------------------------- #
def degeneracy(pts: np.ndarray) -> Tuple[float, float]:
    """(lambda2/lambda1, lambda3/lambda2). Near 1.0 means the axes are arbitrary."""
    x = pts[:, :3] - pts[:, :3].mean(axis=0)
    evals = np.linalg.eigvalsh((x.T @ x) / max(len(x) - 1, 1))
    lam = np.sort(np.maximum(evals, 1e-12))[::-1]
    return float(lam[1] / lam[0]), float(lam[2] / lam[1])


def frame_disagreement(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees.

    This is the number that matters: if the same object seen from two
    viewpoints canonicalises to frames 60 degrees apart, the canonicalisation
    has failed at its one job.
    """
    dR = R_a @ R_b.T
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def yaw_disagreement(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Disagreement about heading only, in degrees, folded to [0, 180]."""
    ya = np.arctan2(R_a[1, 0], R_a[0, 0])
    yb = np.arctan2(R_b[1, 0], R_b[0, 0])
    d = np.degrees(ya - yb) % 360.0
    return float(min(d, 360.0 - d))


def simulate_occlusion(pts: np.ndarray, sensor: np.ndarray,
                       keep: float = 0.5) -> np.ndarray:
    """Keep only the points facing a hypothetical sensor position.

    Crude but honest stand-in for partial visibility: sort by distance to the
    sensor and keep the nearest fraction. Real occlusion is ray-based, but the
    effect on PCA - seeing one side of an object rather than all of it - is the
    same, and this needs no ray casting.
    """
    d = np.linalg.norm(pts[:, :3] - sensor, axis=1)
    k = max(int(len(pts) * keep), 8)
    return pts[np.argsort(d)[:k]]
