"""
Ground removal by sector-wise plane fitting (a lean Patchwork).

Why not one global plane: in Phase 2 we measured a real KITTI road at +0.24 deg
pitch and -0.31 deg roll, and a single plane fitted inside 30 m had 9.1 cm mean
error out at 20-40 m -- worse than assuming a constant height. Roads crown, dip
and slope, so the fit has to be local.

Why polar sectors specifically: the bins then grow with range exactly as point
density falls, so every sector has a comparable number of points to fit to.
A Cartesian grid gives you cells with 200 points near the bumper and 2 at 40 m.

Everything here is numba-jitted and parallel over sectors. On a full KITTI scan
this runs in a few milliseconds on CPU, which matters because it sits in front
of the network on the critical path.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _fit_plane(xs, ys, zs, n):
    """Least-squares z = ax + by + c over n points. Returns (a, b, c, ok)."""
    if n < 3:
        return 0.0, 0.0, 0.0, False
    sx = sy = sz = sxx = sxy = syy = sxz = syz = 0.0
    for i in range(n):
        x = xs[i]; y = ys[i]; z = zs[i]
        sx += x; sy += y; sz += z
        sxx += x * x; sxy += x * y; syy += y * y
        sxz += x * z; syz += y * z
    fn = float(n)
    # normal equations for [a, b, c]
    m00 = sxx; m01 = sxy; m02 = sx
    m11 = syy; m12 = sy
    m22 = fn
    det = (m00 * (m11 * m22 - m12 * m12)
           - m01 * (m01 * m22 - m12 * m02)
           + m02 * (m01 * m12 - m11 * m02))
    if abs(det) < 1e-12:
        return 0.0, 0.0, sz / fn, True          # degenerate -> flat at mean z
    i00 = (m11 * m22 - m12 * m12) / det
    i01 = -(m01 * m22 - m12 * m02) / det
    i02 = (m01 * m12 - m11 * m02) / det
    i11 = (m00 * m22 - m02 * m02) / det
    i12 = -(m00 * m12 - m01 * m02) / det
    i22 = (m00 * m11 - m01 * m01) / det
    a = i00 * sxz + i01 * syz + i02 * sz
    b = i01 * sxz + i11 * syz + i12 * sz
    c = i02 * sxz + i12 * syz + i22 * sz
    return a, b, c, True


@njit(parallel=True, cache=True, fastmath=True)
def _sector_ground(pts, order_by_sec, starts, n_sec, seed_frac, thresh,
                   max_slope, sensor_h, is_ground, ground_z,
                   sec_slope, sec_rough, sec_h, sec_n):
    """One plane per sector, seeded from the lowest points in that sector.

    Points are pre-sorted by sector so each sector is a contiguous slice of
    `order_by_sec`. The obvious implementation -- scanning all N points once per
    sector -- is O(N * n_sec); at 121k points and 1728 sectors that is 209M
    iterations and measured at 80 ms. This is O(N) after one sort, ~4 ms.
    """
    for s in prange(n_sec):
        lo, hi = starts[s], starts[s + 1]
        cnt = hi - lo
        if cnt < 8:
            continue

        idx = order_by_sec[lo:hi]
        zs_all = np.empty(cnt, np.float64)
        for j in range(cnt):
            zs_all[j] = pts[idx[j], 2]

        # seed set: the lowest `seed_frac` of points, which are almost all road
        order = np.argsort(zs_all)
        nseed = max(int(cnt * seed_frac), 3)
        if nseed > cnt:
            nseed = cnt

        # reject a seed set straddling a huge z range - that means the sector is
        # mostly wall, not road, and fitting a plane to it is meaningless
        zlo = zs_all[order[0]]
        zhi = zs_all[order[nseed - 1]]
        if zhi - zlo > 1.0:
            nseed = max(nseed // 2, 3)

        xs = np.empty(nseed, np.float64)
        ys = np.empty(nseed, np.float64)
        zs = np.empty(nseed, np.float64)
        for j in range(nseed):
            i = idx[order[j]]
            xs[j] = pts[i, 0]; ys[j] = pts[i, 1]; zs[j] = pts[i, 2]

        a, b, c, ok = _fit_plane(xs, ys, zs, nseed)
        if not ok:
            continue

        # a near-vertical "ground" plane is a bad fit; fall back to flat
        slope = np.sqrt(a * a + b * b)
        if slope > max_slope:
            a = 0.0; b = 0.0
            c = 0.0
            for j in range(nseed):
                c += zs[j]
            c /= nseed

        # sanity: ground should sit near -sensor_h, not metres away
        if c > -sensor_h + 1.5 or c < -sensor_h - 2.0:
            a = 0.0; b = 0.0; c = -sensor_h

        # --- terrain statistics, essentially free ----------------------- #
        # The plane fit already yields the surface gradient; slope was being
        # computed for a sanity check and then discarded. Roughness is the RMS
        # residual of the sector's own ground points about that plane, which
        # costs one accumulation inside a loop already running. These two plus
        # the inter-sector step are exactly the features the traversability
        # literature scores drivability from.
        sec_slope[s] = np.sqrt(a * a + b * b)
        resid = 0.0
        ng = 0
        mx = 0.0
        my = 0.0
        for j in range(cnt):
            i = idx[j]
            gz = a * pts[i, 0] + b * pts[i, 1] + c
            ground_z[i] = gz
            d = pts[i, 2] - gz
            if d < thresh:
                is_ground[i] = True
                resid += d * d
                mx += pts[i, 0]
                my += pts[i, 1]
                ng += 1
        sec_n[s] = ng
        if ng > 2:
            sec_rough[s] = np.sqrt(resid / ng)
            # Height AT THE SECTOR, not at the origin. `c` is the plane's value
            # at x=y=0, which for a sector 40 m away is an extrapolation across
            # the whole scene: with a 2 degree slope that is over a metre of
            # error, and differencing those between sectors measures
            # extrapolation, not steps. Evaluating at the sector's own centroid
            # gives the local surface height, which is what a step is a
            # difference of.
            mx /= ng
            my /= ng
            sec_h[s] = a * mx + b * my + c


def remove_ground(pts: np.ndarray,
                  n_radial: int = 24,
                  n_azimuth: int = 72,
                  max_range: float = 70.0,
                  seed_frac: float = 0.25,
                  thresh: float = 0.25,
                  max_slope: float = 0.35,
                  sensor_h: float = 1.73):
    """Split a scan into ground and non-ground.

    Returns (is_ground, height_above_ground, sector_stats).

    height_above_ground is a genuinely useful, rotation-invariant network input
    -- far better than raw z, which is measured from the laser rather than the
    road.

    sector_stats carries slope, roughness, fitted height and ground-point count
    per polar sector, plus the sector index of every point. Those are the inputs
    to terrain.py's drivability scoring and they cost effectively nothing here,
    because the plane fit that produces them already runs.
    """
    xy = pts[:, :2].astype(np.float64)
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    th = np.arctan2(xy[:, 1], xy[:, 0])

    # radial bins grow with range: sqrt spacing keeps sector areas comparable
    rn = np.clip(np.sqrt(r / max_range), 0.0, 0.999)
    ri = (rn * n_radial).astype(np.int64)
    ai = np.clip(((th + np.pi) / (2 * np.pi) * n_azimuth).astype(np.int64),
                 0, n_azimuth - 1)
    sec = ri * n_azimuth + ai
    sec[r > max_range] = -1

    p = np.ascontiguousarray(pts[:, :3].astype(np.float64))
    is_ground = np.zeros(len(p), np.bool_)
    ground_z = np.full(len(p), -sensor_h, np.float64)

    n_sec = n_radial * n_azimuth
    # bucket points by sector once: counting sort into a CSR layout, so each
    # sector below is a contiguous slice instead of a full scan of the cloud
    valid = sec >= 0
    sec_v = np.where(valid, sec, n_sec)          # out-of-range -> overflow bin
    counts = np.bincount(sec_v, minlength=n_sec + 1)
    starts = np.zeros(n_sec + 2, np.int64)
    np.cumsum(counts, out=starts[1:])
    order_by_sec = np.argsort(sec_v, kind="stable").astype(np.int64)

    sec_slope = np.zeros(n_sec)
    sec_rough = np.zeros(n_sec)
    sec_h = np.full(n_sec, -sensor_h)
    sec_n = np.zeros(n_sec, np.int64)

    _sector_ground(p, order_by_sec, starts, n_sec, seed_frac, thresh,
                   max_slope, sensor_h, is_ground, ground_z,
                   sec_slope, sec_rough, sec_h, sec_n)

    stats = {"slope": sec_slope, "rough": sec_rough, "h": sec_h, "n": sec_n,
             "sec": sec, "n_radial": n_radial, "n_azimuth": n_azimuth,
             "max_range": max_range}
    return is_ground, (p[:, 2] - ground_z), stats
