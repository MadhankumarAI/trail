"""
Drivable-surface analysis from the statistics ground removal already produces.

THE ARGUMENT
------------
Ground segmentation is a solved geometric problem -- Patchwork++ reaches 96.5%
F1 and GroundGrid 94.8% IoU at 171 Hz, both without learning -- and the
traversability literature scores drivability from three geometric quantities:
slope, roughness, and step height.

All three fall out of a plane fit. `ground.py` already runs one least-squares
fit per polar sector to find the ground, producing `z = ax + by + c`. From that:

    slope      atan(sqrt(a^2 + b^2))            already computed, was discarded
    roughness  RMS residual of the sector's own ground points about its plane
    step       height difference between adjacent sectors' fitted planes

Measured marginal cost of keeping all three: 2.7 ms on a full 121k-point sweep
(33.4 ms against 30.7 ms). Drivability is a byproduct of ground removal, not a
new stage -- the same shape as replacing the learned T-Net with a closed-form
eigendecomposition.

WHERE THIS IS AND ISN'T ENOUGH
------------------------------
Urban driving, which is what PS 26053 describes, is geometry-dominated: a road
is flat, smooth and continuous with where you already are. Learning earns its
keep off-road, where grass, mud and gravel are geometrically identical and only
reflectance separates them.

The hard urban case is road versus sidewalk. Both are flat and smooth; the only
thing between them is a 12 cm kerb. That is a step discontinuity, and it is the
reason `ground_thresh` matters: at the pipeline's default 0.25 m the kerb is
*inside* the ground band, so road and pavement merge into one surface and the
distinction the problem statement asks for cannot be made. Kerb detection here
works on the inter-sector step, not the point-to-plane threshold, so it sees a
12 cm rise that ground removal deliberately ignores.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Thresholds from vehicle dynamics rather than tuning. A passenger car manages
# roughly a 15 degree grade; 10 cm is a typical kerb-mount limit; 5 cm RMS is
# broken tarmac.
MAX_SLOPE_DEG = 15.0
MAX_STEP_M = 0.10
MAX_ROUGH_M = 0.05

# Drivability classes
DRIVABLE, MARGINAL, NON_DRIVABLE, UNKNOWN = 0, 1, 2, 3
NAMES = ["drivable", "marginal", "non-drivable", "unknown"]


@njit(cache=True)
def _sector_step(sec_h, sec_n, n_radial, n_azimuth, out_step):
    """Largest height difference to any adjacent sector.

    Neighbours are the four sectors sharing an edge in the polar grid. Azimuth
    wraps -- sector 0 and sector n_azimuth-1 are adjacent, and forgetting that
    leaves a seam of false kerbs along the +x axis.
    """
    for ri in range(n_radial):
        for ai in range(n_azimuth):
            s = ri * n_azimuth + ai
            if sec_n[s] < 5:
                continue
            best = 0.0
            for d in range(4):
                if d == 0:
                    rj, aj = ri - 1, ai
                elif d == 1:
                    rj, aj = ri + 1, ai
                elif d == 2:
                    rj, aj = ri, (ai - 1) % n_azimuth      # wrap
                else:
                    rj, aj = ri, (ai + 1) % n_azimuth
                if rj < 0 or rj >= n_radial:
                    continue
                t = rj * n_azimuth + aj
                if sec_n[t] < 5:
                    continue
                diff = abs(sec_h[s] - sec_h[t])
                if diff > best:
                    best = diff
            out_step[s] = best


def sector_features(stats: dict) -> dict:
    """Per-sector slope (deg), roughness (m), step (m), and validity."""
    nr, na = stats["n_radial"], stats["n_azimuth"]
    step = np.zeros(nr * na)
    _sector_step(stats["h"], stats["n"], nr, na, step)
    return {
        "slope_deg": np.degrees(np.arctan(stats["slope"])),
        "rough": stats["rough"],
        "step": step,
        "h": stats["h"],
        "n": stats["n"],
        "valid": stats["n"] >= 5,
    }


def classify_sectors(feat: dict,
                     max_slope_deg: float = MAX_SLOPE_DEG,
                     max_step: float = MAX_STEP_M,
                     max_rough: float = MAX_ROUGH_M) -> np.ndarray:
    """Drivability per sector.

    Marginal rather than binary: a surface that violates one criterion mildly is
    not the same as one that violates all three, and a planner wants the
    difference. Anything with too few ground points is unknown, not drivable --
    absence of evidence is not evidence of road.
    """
    out = np.full(len(feat["slope_deg"]), UNKNOWN, np.uint8)
    v = feat["valid"]
    bad = ((feat["slope_deg"] > max_slope_deg).astype(np.int8)
           + (feat["step"] > max_step).astype(np.int8)
           + (feat["rough"] > max_rough).astype(np.int8))
    out[v & (bad == 0)] = DRIVABLE
    out[v & (bad == 1)] = MARGINAL
    out[v & (bad >= 2)] = NON_DRIVABLE
    return out


def point_labels(stats: dict, is_ground: np.ndarray,
                 sector_cls: np.ndarray) -> np.ndarray:
    """Lift sector drivability onto points. Non-ground points are unknown."""
    sec = stats["sec"]
    out = np.full(len(sec), UNKNOWN, np.uint8)
    ok = is_ground & (sec >= 0)
    out[ok] = sector_cls[sec[ok]]
    return out


def kerb_sectors(feat: dict, lo: float = 0.06, hi: float = 0.30) -> np.ndarray:
    """Sectors whose step looks like a kerb rather than a wall or noise.

    A kerb is a small, sharp, *bounded* rise next to an otherwise smooth
    surface. The upper bound is what separates it from a wall; requiring low
    local roughness is what separates it from vegetation, which also produces
    height differences but is rough on both sides.
    """
    return (feat["valid"] & (feat["step"] >= lo) & (feat["step"] <= hi)
            & (feat["rough"] < 0.12) & (feat["slope_deg"] < 20.0))


def analyse(pts: np.ndarray, is_ground: np.ndarray, stats: dict, **kw) -> dict:
    """Everything above, in one call."""
    feat = sector_features(stats)
    sec_cls = classify_sectors(feat, **kw)
    return {
        "feat": feat,
        "sector_cls": sec_cls,
        "point_cls": point_labels(stats, is_ground, sec_cls),
        "kerb": kerb_sectors(feat),
    }
