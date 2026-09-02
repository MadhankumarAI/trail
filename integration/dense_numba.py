"""Fused numba scatter for the dense 2.5D tiers.

WHY THIS EXISTS

The numpy version costs 141 ms a frame, and almost none of it is arithmetic.
Scattering 103k points into four tiers with seven accumulators is about three
million adds -- microseconds of real work. The time goes on twenty-eight
separate bincount passes, each allocating its own output, plus eight
ufunc.at calls, plus a windowed add-back per accumulator per tier. Every pass
re-reads the point arrays and re-walks the same indices.

One kernel does the whole thing in a single pass over the points: compute a
point's cell once per tier, then update all nine values at that cell while the
cache line is hot.

PARALLELISM WITHOUT ATOMICS

Two points in the same sweep can land in the same cell, so a naive prange over
points races on the accumulators. numba's CPU target has no array atomics, and
the usual fix -- per-thread private buffers -- would allocate four copies of a
million-cell map per frame, which costs more than it saves.

So the parallel axis is the TIER. Tiers own disjoint memory, so no two threads
ever touch the same cell, and the loop needs no synchronisation at all. Four
tiers gives four-way parallelism, which is what there is to have.

LAYOUT

Tiers have different sizes, and numba does not take ragged arrays, so all of
them live in one flat buffer with an offset table:

    off[t]      where tier t's cells begin
    n[t]        tier t is n[t] x n[t]
    res[t]      its cell size
    org[t, :]   the world position of its centre

acc is (9, total_cells): the seven summed accumulators plus zmax and gmin,
which are kept in the same buffer so the kernel writes to one array.
"""

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                   # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or callable(a[0]) is False else a[0]
    prange = range

# rows of `acc`
N, NG, GSUM, GSQ, GW, GWZ, GWZ2, ZMAX, GMIN = range(9)
NROW = 9


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def scatter(x, y, z, isg, w, org, res, n, off, acc):
    """Fold one sweep into every tier, one pass over the points per tier.

    Parallel over tiers only: they own disjoint slices of `acc`, so there is
    no race and no atomic. Within a tier the loop is serial, which is what
    makes `+=` on a shared cell correct.
    """
    ntier = res.shape[0]
    npts = x.shape[0]
    for t in prange(ntier):
        r = res[t]
        half = n[t] // 2
        base = off[t]
        nt = n[t]
        ox = org[t, 0]
        oy = org[t, 1]
        for p in range(npts):
            ix = int(np.floor((x[p] - ox) / r)) + half
            if ix < 0 or ix >= nt:
                continue
            iy = int(np.floor((y[p] - oy) / r)) + half
            if iy < 0 or iy >= nt:
                continue
            c = base + ix * nt + iy
            zp = z[p]

            acc[N, c] += 1.0
            if zp > acc[ZMAX, c]:
                acc[ZMAX, c] = zp
            if isg[p]:
                acc[NG, c] += 1.0
                acc[GSUM, c] += zp
                acc[GSQ, c] += zp * zp
                wp = w[p]
                acc[GW, c] += wp
                acc[GWZ, c] += wp * zp
                acc[GWZ2, c] += wp * zp * zp
                if zp < acc[GMIN, c]:
                    acc[GMIN, c] = zp


@njit(cache=True, fastmath=True, nogil=True)
def align_offset(x, y, z, isg, org, res, n, off, acc, t, min_cells):
    """Median height difference between this sweep and the tier it overlaps.

    Collects the per-point differences into a buffer and returns it; the
    caller takes the median. Written as a kernel because the indexing is the
    expensive part, not the median.
    """
    r = res[t]
    half = n[t] // 2
    base = off[t]
    nt = n[t]
    ox = org[t, 0]
    oy = org[t, 1]
    out = np.empty(x.shape[0])
    k = 0
    for p in range(x.shape[0]):
        if not isg[p]:
            continue
        ix = int(np.floor((x[p] - ox) / r)) + half
        if ix < 0 or ix >= nt:
            continue
        iy = int(np.floor((y[p] - oy) / r)) + half
        if iy < 0 or iy >= nt:
            continue
        c = base + ix * nt + iy
        ng = acc[NG, c]
        if ng >= 3.0:
            out[k] = acc[GSUM, c] / ng - z[p]
            k += 1
    return out[:k]


@njit(cache=True, nogil=True)
def shift_tier(acc, off, n, t, dx, dy):
    """Recentre one tier in place, keeping the overlap.

    np.roll allocates a copy of every row; this walks the destination once and
    blanks what came into view, touching each cell exactly once.
    """
    nt = n[t]
    base = off[t]
    for row in range(NROW):
        blank = -np.inf if row == ZMAX else (np.inf if row == GMIN else 0.0)
        if dx > 0:
            xs = range(0, nt)
        else:
            xs = range(nt - 1, -1, -1)
        for i in xs:
            si = i + dx
            for j in (range(0, nt) if dy > 0 else range(nt - 1, -1, -1)):
                sj = j + dy
                d = base + i * nt + j
                if 0 <= si < nt and 0 <= sj < nt:
                    acc[row, d] = acc[row, base + si * nt + sj]
                else:
                    acc[row, d] = blank


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def scatter_raw(pts, lab, R, tv, trust2, gnd_lut, s0sq, srsq,
                dz, org, res, n, off, acc):
    """Gate, transform, weight and scatter -- one kernel, no temporaries.

    The previous split cost more in numpy than the kernel it fed: a norm over
    every point (9.1 ms), three boolean-mask copies (3.8 ms), the pose
    transform (3.4 ms) and three ascontiguousarray calls, against 3.5 ms of
    actual scattering. Every one of those allocated an array whose only
    purpose was to be read once.

    So the kernel reads the raw sweep. The range gate is on the SQUARED
    distance, which avoids a sqrt for every point when only the ones inside
    the gate need a real range for their weight.

    The pose transform is recomputed per tier rather than shared. That is four
    times the arithmetic and still nothing -- nine multiplies per point per
    tier against a memory round trip it would otherwise cost -- and it keeps
    the tiers independent, which is what lets them run in parallel without a
    lock.
    """
    ntier = res.shape[0]
    npts = pts.shape[0]
    for t in prange(ntier):
        r = res[t]
        half = n[t] // 2
        base = off[t]
        nt = n[t]
        ox = org[t, 0]
        oy = org[t, 1]
        for p in range(npts):
            px = pts[p, 0]
            py = pts[p, 1]
            d2 = px * px + py * py
            if d2 > trust2:
                continue
            pz = pts[p, 2]

            wx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + tv[0]
            ix = int(np.floor((wx - ox) / r)) + half
            if ix < 0 or ix >= nt:
                continue
            wy = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + tv[1]
            iy = int(np.floor((wy - oy) / r)) + half
            if iy < 0 or iy >= nt:
                continue
            wz = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + tv[2] + dz

            c = base + ix * nt + iy
            acc[N, c] += 1.0
            if wz > acc[ZMAX, c]:
                acc[ZMAX, c] = wz
            if gnd_lut[lab[p]]:
                w = 1.0 / (s0sq + srsq * d2)
                acc[NG, c] += 1.0
                acc[GSUM, c] += wz
                acc[GSQ, c] += wz * wz
                acc[GW, c] += w
                acc[GWZ, c] += w * wz
                acc[GWZ2, c] += w * wz * wz
                if wz < acc[GMIN, c]:
                    acc[GMIN, c] = wz


@njit(cache=True, fastmath=True, nogil=True)
def align_raw(pts, lab, R, tv, trust2, gnd_lut, org, res, n, off, acc, t):
    """Height differences against tier t, straight from the raw sweep."""
    r = res[t]
    half = n[t] // 2
    base = off[t]
    nt = n[t]
    ox = org[t, 0]
    oy = org[t, 1]
    out = np.empty(pts.shape[0])
    k = 0
    for p in range(pts.shape[0]):
        if not gnd_lut[lab[p]]:
            continue
        px = pts[p, 0]
        py = pts[p, 1]
        if px * px + py * py > trust2:
            continue
        pz = pts[p, 2]
        wx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + tv[0]
        ix = int(np.floor((wx - ox) / r)) + half
        if ix < 0 or ix >= nt:
            continue
        wy = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + tv[1]
        iy = int(np.floor((wy - oy) / r)) + half
        if iy < 0 or iy >= nt:
            continue
        c = base + ix * nt + iy
        ng = acc[NG, c]
        if ng >= 3.0:
            wz = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + tv[2]
            out[k] = acc[GSUM, c] / ng - wz
            k += 1
    return out[:k]


# --------------------------------------------------------------------------
# Circular-buffer tiers. The map does not move; only its origin index does.
# --------------------------------------------------------------------------
#
# shift_tier above physically relocates every cell, which is 980k cells across
# nine rows per frame and measured ~40 ms -- five times the cost of the scatter
# it exists to serve. grid_map does not do that: GridMap::move only advances
# startIndex_ and calls clearRows/clearCols on the strip that came into view,
# leaving the overlap exactly where it lies.
#
# Same idea here. Each tier keeps a start offset, a cell's buffer slot is
# (i + start) mod n, and a move clears dx rows instead of copying n rows. At
# 5 cm and 0.9 m of travel that is 18 rows touched rather than 480.

@njit(cache=True, nogil=True)
def clear_strip(acc, off, n, t, sx0, dx, sy0, dy):
    """Blank the rows and columns a move brought into view.

    Indices are given in BUFFER space and wrap, because the strip that becomes
    visible is contiguous on the map but need not be contiguous in memory.
    """
    nt = n[t]
    base = off[t]
    for row in range(NROW):
        blank = -np.inf if row == ZMAX else (np.inf if row == GMIN else 0.0)
        for k in range(dx):
            i = (sx0 + k) % nt
            for j in range(nt):
                acc[row, base + i * nt + j] = blank
        for k in range(dy):
            j = (sy0 + k) % nt
            for i in range(nt):
                acc[row, base + i * nt + j] = blank


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def scatter_ring(pts, lab, R, tv, trust2, gnd_lut, s0sq, srsq, dz,
                 org, res, n, off, start, acc):
    """scatter_raw, but addressing a circular buffer."""
    ntier = res.shape[0]
    npts = pts.shape[0]
    for t in prange(ntier):
        r = res[t]
        half = n[t] // 2
        base = off[t]
        nt = n[t]
        ox = org[t, 0]
        oy = org[t, 1]
        sx = start[t, 0]
        sy = start[t, 1]
        for p in range(npts):
            px = pts[p, 0]
            py = pts[p, 1]
            d2 = px * px + py * py
            if d2 > trust2:
                continue
            pz = pts[p, 2]

            wx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + tv[0]
            ix = int(np.floor((wx - ox) / r)) + half
            if ix < 0 or ix >= nt:
                continue
            wy = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + tv[1]
            iy = int(np.floor((wy - oy) / r)) + half
            if iy < 0 or iy >= nt:
                continue
            wz = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + tv[2] + dz

            c = base + ((ix + sx) % nt) * nt + ((iy + sy) % nt)
            acc[N, c] += 1.0
            if wz > acc[ZMAX, c]:
                acc[ZMAX, c] = wz
            if gnd_lut[lab[p]]:
                w = 1.0 / (s0sq + srsq * d2)
                acc[NG, c] += 1.0
                acc[GSUM, c] += wz
                acc[GSQ, c] += wz * wz
                acc[GW, c] += w
                acc[GWZ, c] += w * wz
                acc[GWZ2, c] += w * wz * wz
                if wz < acc[GMIN, c]:
                    acc[GMIN, c] = wz


@njit(cache=True, fastmath=True, nogil=True)
def align_ring(pts, lab, R, tv, trust2, gnd_lut, org, res, n, off, start,
               acc, t):
    """align_raw against a circular buffer."""
    r = res[t]
    half = n[t] // 2
    base = off[t]
    nt = n[t]
    ox = org[t, 0]
    oy = org[t, 1]
    sx = start[t, 0]
    sy = start[t, 1]
    out = np.empty(pts.shape[0])
    k = 0
    for p in range(pts.shape[0]):
        if not gnd_lut[lab[p]]:
            continue
        px = pts[p, 0]
        py = pts[p, 1]
        if px * px + py * py > trust2:
            continue
        pz = pts[p, 2]
        wx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + tv[0]
        ix = int(np.floor((wx - ox) / r)) + half
        if ix < 0 or ix >= nt:
            continue
        wy = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + tv[1]
        iy = int(np.floor((wy - oy) / r)) + half
        if iy < 0 or iy >= nt:
            continue
        c = base + ((ix + sx) % nt) * nt + ((iy + sy) % nt)
        ng = acc[NG, c]
        if ng >= 3.0:
            wz = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + tv[2]
            out[k] = acc[GSUM, c] / ng - wz
            k += 1
    return out[:k]
