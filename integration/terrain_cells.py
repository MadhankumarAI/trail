"""
drivability as a LAYER of the 2.5d grid, computed from grid25's own cells.

until now drivability lived on its own polar sector grid, separate from the map
it was meant to annotate. that meant two data structures, two resolutions and a
resampling step between them -- exactly the "alignment errors and data loss"
the problem statement warns about, reintroduced one stage later.

grid25 already stores everything the calculation needs:

    ng, gsum, gsq   count, sum and sum-of-squares of GROUND heights in the cell
                    -> mean height and its standard deviation, directly
    neighbours      cells are on a regular lattice at the fine level, so a step
                    is a difference against ix +/- 1, iy +/- 1

so roughness needs no plane fit at all here: the cell IS the neighbourhood, and
the spread of ground heights inside it is the roughness. slope and step come
from differencing neighbouring cell heights.

and because those accumulators merge across frames (see accumulate.py), the same
code runs on one sweep or on a hundred. that is the point: the far field is
starved of evidence, not of arithmetic.
"""

import numpy as np

import grid25 as g

# thresholds carried over from the polar implementation, where they were swept
# against KITTI ROAD: slope and step from vehicle dynamics, roughness measured.
MAX_SLOPE_DEG = 15.0
MAX_STEP_M = 0.10
# Re-derived for THIS estimator, not inherited. The 2 cm figure came from the
# polar version, where roughness was the MAD of residuals about a fitted plane
# over a whole sector. Here it is the tilt-corrected standard deviation of
# ground heights inside a 40 cm cell, accumulated over many viewpoints -- a
# different quantity on a different scale, and carrying the old number across
# rejected 30% of genuine road.
#
# Swept on seq 00 against SemanticKITTI: road cells must come out drivable,
# cells holding buildings/poles/vehicles must not.
#
#   thr    road recall    false-drivable    F1
#   0.02      69.8%           11.5%        0.798
#   0.03      86.3%           14.3%        0.896
#   0.04      90.5%           17.4%        0.913   <- chosen
#   0.05      92.0%           18.7%        0.919
#   0.10      94.2%           30.1%        0.908
#
# 0.05 scores marginally higher but buys 1.5 pp of recall with 1.3 pp more
# false-drivable, and a false drivable is the dangerous error. Note the
# false-drivable floor is 11.5% even at the tightest setting, so most of it is
# not threshold-driven -- it is cells where an obstacle clips the edge.
MAX_ROUGH_M = 0.040
STEP_D_SF = 8.0e-5          # step tolerance grows with range; roughness does not
MIN_GROUND_PTS = 5          # below this the cell reports unknown, not a class

# Drivability is computed on a COARSER lattice than the map's finest cells.
# A 5 cm cell holds one to three returns - there is no surface statistic to be
# had from that, and requiring enough points would mark the entire map unknown.
# 40 cm is the scale a wheel actually cares about, and it is an exact power-of-
# two coarsening of grid25's own grid, so cells still nest without resampling.
DRIVE_LEVEL = 3             # res0 * 2**3 = 0.40 m

DRIVABLE, MARGINAL, NON_DRIVABLE, UNKNOWN = 0, 1, 2, 3
NAMES = ["drivable", "marginal", "non-drivable", "unknown"]


def _neighbour_delta(ix, iy, h, valid):
    """Largest height difference to any of the four edge neighbours.

    Cells are sparse, so neighbours are found by hashing (ix, iy) rather than
    by indexing a dense array -- the map is mostly empty and densifying it at
    5 cm over a 90 m radius would be 13 million cells to hold a few tens of
    thousands.
    """
    key = g._pack(ix, iy)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    hs = np.where(valid[order], h[order], np.nan)

    out = np.zeros(len(ix))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = g._pack(ix + dx, iy + dy)
        pos = np.searchsorted(ks, q)
        pos = np.clip(pos, 0, len(ks) - 1)
        hit = ks[pos] == q
        d = np.abs(np.where(hit, hs[pos], np.nan) - h)
        out = np.fmax(out, np.nan_to_num(d, nan=0.0))
    return out


def coarsen(c, level=DRIVE_LEVEL):
    """Merge fine cells up to the drivability lattice. Exact, not resampled:
    a parent index is a right shift, and every accumulator is associative."""
    from accumulate import merge_ext
    px, py = c["ix"] >> level, c["iy"] >> level
    m, o, st = merge_ext(c, g._pack(px, py))
    m["ix"], m["iy"] = px[o][st], py[o][st]
    return m


def cell_drivability(c, sensor_xy=(0.0, 0.0), res=g.res0 * (1 << DRIVE_LEVEL),
                     max_slope_deg=MAX_SLOPE_DEG, max_step=MAX_STEP_M,
                     max_rough=MAX_ROUGH_M, step_d_sf=STEP_D_SF,
                     min_pts=MIN_GROUND_PTS, terms=False):
    """Per-cell drivability cost and class from grid25 accumulators.

    Returns (score, cls, height). Score is the worst normalised violation, so
    1.0 is exactly the vehicle limit and the quantity stays continuous -- a
    planner can use the margin, and it can be smoothed or fused.
    """
    ng = c["ng"]
    ok = ng >= min_pts

    # Prefer the range-weighted accumulators when the map carries them (see
    # accumulate.py): a patch seen once from 40 m must not outvote the same
    # patch seen ten times from 8 m. Falls back to unweighted sums for a plain
    # single-frame quantise(), where every point shares one viewpoint anyway.
    with np.errstate(invalid="ignore", divide="ignore"):
        if "gw" in c:
            wsum = np.maximum(c["gw"], 1e-12)
            mean = np.where(ok, c["gwz"] / wsum, np.nan)
            var = np.where(ok, c["gwz2"] / wsum - mean * mean, np.nan)
        else:
            mean = np.where(ok, c["gsum"] / np.maximum(ng, 1), np.nan)
            var = np.where(ok, c["gsq"] / np.maximum(ng, 1) - mean * mean,
                           np.nan)

    cx = (c["ix"] + 0.5) * res
    cy = (c["iy"] + 0.5) * res
    rng = np.hypot(cx - sensor_xy[0], cy - sensor_xy[1])

    step = _neighbour_delta(c["ix"], c["iy"], np.nan_to_num(mean), ok)
    slope_deg = np.degrees(np.arctan(step / max(res, 1e-6)))

    # The spread of ground heights inside a cell is NOT roughness on its own:
    # a perfectly smooth road at a legal 15 degree grade still varies by ~3 cm
    # across a 40 cm cell. Charging that as roughness condemns every hill, and
    # since slope is scored separately it would be charged twice.
    #
    # The polar version avoided this by fitting a plane and measuring residuals.
    # There is no plane here, but there is no need for one: for a surface of
    # gradient s sampled across a cell of side L, the slope contributes
    # (s*L)^2/12 to the height variance, and s*L is exactly the neighbouring
    # height difference already computed above. So the tilt can be removed
    # analytically, leaving the residual bumpiness a wheel actually feels.
    rough = np.sqrt(np.maximum(var - step * step / 12.0, 0.0))

    step_t = max_step + step_d_sf * rng * rng
    score = np.maximum.reduce([
        rough / max_rough,
        step / np.maximum(step_t, 1e-6),
        slope_deg / max_slope_deg,
    ])
    score = np.where(ok, score, np.nan)

    cls = np.full(len(ng), UNKNOWN, np.uint8)
    good = np.isfinite(score)
    cls[good & (score <= 1.0)] = DRIVABLE
    cls[good & (score > 1.0) & (score <= 2.0)] = MARGINAL
    cls[good & (score > 2.0)] = NON_DRIVABLE
    if terms:
        # the three criteria separately, each normalised the same way as the
        # combined score, so a consumer can see WHICH one condemned a cell --
        # a kerb and a rubble field are both "non-drivable" and want different
        # responses from a planner
        # RAW physical quantities, not divided by our thresholds. Those
        # thresholds are a calibrated decision boundary, not a vehicle limit,
        # so pre-normalising by them would force every consumer to inherit our
        # calibration whether it suits their platform or not.
        return score, cls, mean, {
            "rough_m": np.where(ok, rough, np.nan),
            "step_m": np.where(ok, step, np.nan),
            "slope_deg": np.where(ok, slope_deg, np.nan),
        }
    return score, cls, mean


def summarise(cls, rng, bands=((0, 10), (10, 20), (20, 30), (30, 50))):
    """Class mix by range, as fractions of all cells in the band."""
    out = {}
    for lo, hi in bands:
        m = (rng >= lo) & (rng < hi)
        if m.sum() == 0:
            continue
        tot = m.sum()
        out[f"{lo}-{hi}"] = {
            NAMES[k]: float((cls[m] == k).sum()) / tot
            for k in (DRIVABLE, MARGINAL, NON_DRIVABLE, UNKNOWN)
        } | {"cells": int(tot)}
    return out
