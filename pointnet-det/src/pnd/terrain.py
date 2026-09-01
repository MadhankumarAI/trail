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

# Slope and step come from vehicle dynamics: a passenger car manages roughly a
# 15 degree grade, and 10 cm is a typical kerb-mount limit.
#
# The roughness threshold cannot come from dynamics, because it is a property of
# the *estimator* as much as the terrain. The traversability literature is
# consistently qualitative here - it names slope, roughness and step as the
# features and leaves the numbers to the platform - so this one is measured.
# Swept on KITTI ROAD with MAD scatter: 0.008 -> F1 0.622, 0.016 -> 0.797,
# 0.020 -> 0.798, 0.025 -> 0.787. Flat-topped around 0.020.
#
# It is roughly half the old 0.05 because MAD reads about half of RMS on the
# same data. Changing the estimator without re-deriving the threshold is what
# made the first MAD attempt worse, not better.
MAX_SLOPE_DEG = 15.0
MAX_STEP_M = 0.10
MAX_ROUGH_M = 0.020       # MAD, not RMS - see ground.py

# DISTANCE-SCALED TOLERANCE, after GroundGrid (Steinke et al., RA-L 2024), which
# thresholds cell variance as t = t_min + d_sf * d^2 rather than at a constant.
#
# The reason is measurement, not terrain. Beam divergence and range noise both
# grow with distance, so a genuinely flat road scatters more at 30 m than at
# 5 m. A fixed roughness threshold therefore rejects real road as the range
# grows: measured recall fell from 0.931 in 0-10 m to 0.262 at 20-40 m, which is
# the sensor getting noisier, not the road getting rougher.
#
# Same argument for step: adjacent sector plane heights disagree more at range
# because each plane is fitted to fewer, noisier points.
#
# MEASURED: the idea only half transfers. Sweeping both coefficients on KITTI
# ROAD, scaling the STEP tolerance helps -- far-field F1 0.403 -> 0.439, recall
# 0.284 -> 0.337 at no overall cost -- but scaling ROUGHNESS hurts at every
# setting (overall F1 0.768 -> 0.726 as it rises), so it is off.
#
# The difference from GroundGrid is the quantity being thresholded. Theirs is
# raw cell height variance, which grows with range purely from sensor noise.
# Ours is the RMS residual about a plane that was *fitted to those same points*,
# and the fit already absorbs most of that spread. Scaling it a second time
# over-relaxes and admits genuinely rough non-road. Step has no fit protecting
# it, so it scales as GroundGrid predicts.
ROUGH_D_SF = 0.0          # measured harmful - see note
STEP_D_SF = 8.0e-5        # metres of extra step tolerance per metre^2

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
    # centre range of each radial ring. remove_ground bins radius as
    # sqrt(r/max_range), so invert that to recover metres.
    ring = np.arange(nr)
    r_edge = ((ring + 0.5) / nr) ** 2 * stats["max_range"]
    rng = np.repeat(r_edge, na)
    return {
        "range": rng,
        "slope_deg": np.degrees(np.arctan(stats["slope"])),
        "rough": stats["rough"],
        "step": step,
        "h": stats["h"],
        "n": stats["n"],
        "valid": stats["n"] >= 5,
    }


def drivability_score(feat: dict,
                      max_slope_deg: float = MAX_SLOPE_DEG,
                      max_step: float = MAX_STEP_M,
                      max_rough: float = MAX_ROUGH_M,
                      rough_d_sf: float = ROUGH_D_SF,
                      step_d_sf: float = STEP_D_SF,
                      smooth: bool = True,
                      n_radial: int = 24, n_azimuth: int = 72) -> np.ndarray:
    """Continuous cost, 0 = ideal, 1 = at the limit, >1 = over it.

    Counting how many binary tests a sector fails throws away magnitude and
    chatters: a sector at roughness 0.0199 and one at 0.0201 differ by a tenth
    of a millimetre and land in different classes. Measured on 40 consecutive
    frames, 24.1% of sectors changed class between frames 100 ms apart and 10.5%
    flipped drivable <-> not. A planner re-deciding one cell in ten at 10 Hz is
    a planner that jerks.

    Taking the worst normalised violation keeps the physical meaning -- 1.0 is
    still exactly the vehicle limit -- while making the quantity continuous, so
    it can be smoothed and so a cost-based planner can use the margin rather
    than a hard label.
    """
    d = feat.get("range")
    rough_t = max_rough + (rough_d_sf * d * d if d is not None else 0.0)
    step_t = max_step + (step_d_sf * d * d if d is not None else 0.0)
    sc = np.maximum.reduce([
        feat["rough"] / np.maximum(rough_t, 1e-6),
        feat["step"] / np.maximum(step_t, 1e-6),
        feat["slope_deg"] / max_slope_deg,
    ])
    sc = np.where(feat["valid"], sc, np.nan)

    if smooth:
        # Median over the sector and its four edge neighbours. Terrain is
        # spatially coherent -- road does not alternate with verge every 5
        # degrees of azimuth -- so a lone dissenting sector is far more likely
        # to be a bad fit than a real feature. Measured neighbour disagreement
        # before smoothing: 36.2%.
        g = sc.reshape(n_radial, n_azimuth)
        stack = np.stack([
            g,
            np.roll(g, 1, axis=1), np.roll(g, -1, axis=1),   # azimuth wraps
            np.vstack([g[:1], g[:-1]]),                       # radial, clamped
            np.vstack([g[1:], g[-1:]]),
        ])
        with np.errstate(invalid="ignore"):
            sc = np.nanmedian(stack, axis=0).reshape(-1)
    return sc


def classify_from_score(sc: np.ndarray, marginal_at: float = 1.0,
                        nondriv_at: float = 2.0) -> np.ndarray:
    out = np.full(len(sc), UNKNOWN, np.uint8)
    ok = np.isfinite(sc)
    out[ok & (sc <= marginal_at)] = DRIVABLE
    out[ok & (sc > marginal_at) & (sc <= nondriv_at)] = MARGINAL
    out[ok & (sc > nondriv_at)] = NON_DRIVABLE
    return out


def classify_sectors(feat: dict,
                     max_slope_deg: float = MAX_SLOPE_DEG,
                     max_step: float = MAX_STEP_M,
                     max_rough: float = MAX_ROUGH_M,
                     rough_d_sf: float = ROUGH_D_SF,
                     step_d_sf: float = STEP_D_SF) -> np.ndarray:
    """Drivability per sector, with range-dependent tolerances.

    Marginal rather than binary: a surface that violates one criterion mildly is
    not the same as one that violates all three, and a planner wants the
    difference. Anything with too few ground points is unknown, not drivable --
    absence of evidence is not evidence of road.

    Note that the slope test is kept for off-road generality but does no work in
    the city: sweeping it over 10, 15 and 20 degrees on KITTI ROAD gave
    bit-identical results, because urban roads are flat (measured median slope
    0.94 deg, p95 7.7). Roughness and step carry the whole signal here.
    """
    out = np.full(len(feat["slope_deg"]), UNKNOWN, np.uint8)
    v = feat["valid"]
    d = feat.get("range")
    if d is None:
        rough_t = np.full(len(out), max_rough)
        step_t = np.full(len(out), max_step)
    else:
        rough_t = max_rough + rough_d_sf * d * d
        step_t = max_step + step_d_sf * d * d
    bad = ((feat["slope_deg"] > max_slope_deg).astype(np.int8)
           + (feat["step"] > step_t).astype(np.int8)
           + (feat["rough"] > rough_t).astype(np.int8))
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


class TerrainTracker:
    """Temporal fusion of the drivability cost across frames.

    Without it every frame is computed from scratch and the output chatters:
    measured over 40 consecutive sweeps, 24.1% of sectors changed class between
    frames 100 ms apart and 10.5% flipped drivable <-> not. That is a planner
    re-deciding one cell in ten at 10 Hz.

    The cost is fused with an exponential update rather than recomputed, which
    is the same idea as the Kalman height update in elevation_mapping_cupy: a
    cell's state is evidence accumulated over time, not the last measurement.

    HONEST LIMITATION. Sectors are indexed relative to the vehicle, so fusing by
    index assumes the terrain at "3 m ahead, 10 degrees left" is the same
    terrain it was 100 ms ago. At 15 km/h that is 0.4 m of drift per frame -
    small against a near sector, not small against a far one, and wrong through
    a turn. Doing it properly means fusing in a world-anchored frame using the
    ego pose, which is exactly what the 2.5D map is for; this belongs there once
    that exists. Until then this is a real improvement with a known bias, not a
    finished solution.
    """

    def __init__(self, alpha: float = 0.4, hysteresis: float = 0.15):
        self.alpha = alpha
        self.hyst = hysteresis
        self.score = None
        self.cls = None

    def update(self, sc: np.ndarray) -> np.ndarray:
        """Fuse this frame's cost into the running estimate."""
        if self.score is None or self.score.shape != sc.shape:
            self.score = sc.copy()
        else:
            fresh = np.isfinite(sc)
            stale = ~np.isfinite(self.score)
            blend = self.alpha * sc + (1.0 - self.alpha) * self.score
            self.score = np.where(fresh & ~stale, blend,
                                  np.where(fresh, sc, self.score))
        return self.score

    def classify(self, sc: np.ndarray) -> np.ndarray:
        """Band the fused cost, with hysteresis so a sector sitting exactly on a
        threshold does not toggle every frame. Crossing needs `hyst` of margin
        in the direction of change."""
        new = classify_from_score(sc)
        if self.cls is None:
            self.cls = new
            return new
        keep = np.zeros(len(sc), bool)
        ok = np.isfinite(sc)
        for thr in (1.0, 2.0):
            near = ok & (np.abs(sc - thr) < self.hyst)
            keep |= near
        out = np.where(keep, self.cls, new).astype(np.uint8)
        self.cls = out
        return out


def analyse(pts: np.ndarray, is_ground: np.ndarray, stats: dict,
            smooth: bool = True, score: np.ndarray | None = None, **kw) -> dict:
    """Everything above, in one call.

    `score` lets a caller pass a temporally fused cost in place of this frame's
    own, so tracking lives in one place instead of being reimplemented per
    consumer.
    """
    feat = sector_features(stats)
    sc = score if score is not None else drivability_score(
        feat, smooth=smooth,
        n_radial=stats["n_radial"], n_azimuth=stats["n_azimuth"], **kw)
    sec_cls = classify_from_score(sc)
    return {
        "feat": feat,
        "score": sc,
        "sector_cls": sec_cls,
        "point_cls": point_labels(stats, is_ground, sec_cls),
        "kerb": kerb_sectors(feat),
    }
