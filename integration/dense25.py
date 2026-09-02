"""Dense robot-centric 2.5D accumulation. No sorting anywhere.

WHY THIS REPLACES THE SPARSE HASH
---------------------------------
The sparse version kept cells in a hash over a 90 m radius at 5 cm -- 296,313
cells by frame 18 -- and folded each sweep in by concatenating, sorting and
re-reducing every accumulator. Measured on seq 00: 181 ms to accumulate and
174 ms to foveate, 55% of a 648 ms frame. Only 8.5 ms of that was the sort
itself; the rest was fifteen gathers over 341k elements to update 45k.

The reference implementation for this problem (Miki et al., "Elevation Mapping
for Locomotion and Navigation using GPU", and leggedrobotics/elevation_mapping_cupy)
does not do that. It keeps a DENSE, FIXED-SIZE, ROBOT-CENTRIC grid -- 10 x 10 m
at 4 cm, 62,500 cells -- and scatters points into it with atomics. No sort, no
hash, no growth. They report 6.9 ms per update on a Jetson Xavier.

Dense works there because the map is small. It is small because it does not
need to be fine everywhere -- which is exactly what this problem statement
asks for: 5 cm within 10 m, degrading with range. So the map here is a stack
of dense tiers, each a fixed-size array covering a fixed radius at a fixed
resolution:

    tier 0    5 cm over  +/- 12 m      480 x 480     230,400 cells
    tier 1   10 cm over  +/- 25 m      500 x 500     250,000 cells
    tier 2   20 cm over  +/- 50 m      500 x 500     250,000 cells
    tier 3   40 cm over  +/- 100 m     500 x 500     250,000 cells

A point lands in every tier whose radius covers it, and a query reads the
finest tier that has evidence. Foveation costs nothing at query time because
the tiers ARE the foveation -- there is no merge step to run.

WHY THERE IS NO SORT
--------------------
With a dense grid, a cell has a linear index, so accumulating is
np.bincount(idx, weights=...) -- one pass, O(n), no ordering. The sparse
version had to sort because it could not index directly into anything.

WHAT MOVES AND WHAT DOES NOT
----------------------------
The grid is robot-centric and shifts with the vehicle by whole cells, exactly
as grid_map's move() does: the overlap is preserved and only the strip that
comes into view is cleared. The cells stay world-aligned to within one cell,
so a patch does not drift between frames.

The accumulators are the same ones grid25 keeps, so the drivability code reads
this without changing.
"""

import numpy as np

# (resolution m, half-extent m). Each tier is a dense square array.
TIERS = ((0.05, 12.0), (0.10, 25.0), (0.20, 50.0), (0.40, 100.0))

# Per-cell accumulators, all associative under addition.
#
# zsum and zsq were dropped: every consumer reads the GROUND statistics, and
# the weighted trio (gw, gwz, gwz2) is what drivability actually uses, with
# gsum/gsq kept for the unweighted variance to_gridmap publishes. Each field
# is one more bincount per tier per frame -- nine of them across four tiers
# was 67 ms, and two were feeding nothing.
_SUM = ("n", "ng", "gsum", "gsq", "gw", "gwz", "gwz2")

SIGMA0, SIGMA_R = 0.02, 0.005


class Tier:
    """One dense level: fixed resolution, fixed extent, robot-centric."""

    def __init__(self, res, half):
        self.res = float(res)
        self.n = int(round(2 * half / res))
        self.half = self.n * res / 2.0
        self.origin = np.zeros(2)          # world position of the grid centre
        self.a = {k: np.zeros(self.n * self.n) for k in _SUM}
        self.zmax = np.full(self.n * self.n, -np.inf)
        self.gmin = np.full(self.n * self.n, np.inf)

    # -- geometry ----------------------------------------------------- #
    def index(self, x, y):
        """Linear cell index for world points, and a mask of what is inside."""
        ix = np.floor((x - self.origin[0]) / self.res).astype(np.int64) + self.n // 2
        iy = np.floor((y - self.origin[1]) / self.res).astype(np.int64) + self.n // 2
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        return ix * self.n + iy, ok

    def centres(self):
        """World (x, y) of every cell centre, as flat arrays."""
        g = (np.arange(self.n) - self.n // 2 + 0.5) * self.res
        gx, gy = np.meshgrid(g, g, indexing="ij")
        return gx.ravel() + self.origin[0], gy.ravel() + self.origin[1]

    def move(self, pos):
        """Recentre on the vehicle, keeping the overlap.

        Shifts by whole cells only, so a world patch keeps its cell to within
        half a cell and does not smear as the vehicle moves. Rows and columns
        that come into view are cleared, not carried over from the far side.
        """
        d = np.rint((np.asarray(pos, float) - self.origin) / self.res).astype(int)
        if not d.any():
            return
        for key in list(self.a) + ["zmax", "gmin"]:
            buf = self.a[key] if key in self.a else getattr(self, key)
            m = buf.reshape(self.n, self.n)
            m = np.roll(m, (-d[0], -d[1]), axis=(0, 1))
            blank = -np.inf if key == "zmax" else (np.inf if key == "gmin" else 0.0)
            if d[0] > 0:
                m[-d[0]:, :] = blank
            elif d[0] < 0:
                m[:-d[0], :] = blank
            if d[1] > 0:
                m[:, -d[1]:] = blank
            elif d[1] < 0:
                m[:, :-d[1]] = blank
            if key in self.a:
                self.a[key] = m.ravel()
            else:
                setattr(self, key, m.ravel())
        self.origin = self.origin + d * self.res

    # -- ingest ------------------------------------------------------- #
    def add(self, x, y, z, isg, w):
        """Scatter one sweep in. bincount, not sort -- this is the whole point.

        Binned over the WINDOW the sweep touches, not the whole tier. A sweep
        is gated to 20 m, so at 40 cm it covers a 100 x 100 patch of a
        500 x 500 tier; binning over the tier meant zeroing 250k cells to hold
        points that fit in 10k. Each bincount is cheap, but there is one per
        accumulator per tier and the zeroing dominated.
        """
        ix = np.floor((x - self.origin[0]) / self.res).astype(np.int64) + self.n // 2
        iy = np.floor((y - self.origin[1]) / self.res).astype(np.int64) + self.n // 2
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        if not ok.any():
            return
        ix, iy, z, isg, w = ix[ok], iy[ok], z[ok], isg[ok], w[ok]

        x0, x1 = int(ix.min()), int(ix.max()) + 1
        y0, y1 = int(iy.min()), int(iy.max()) + 1
        wx, wy = x1 - x0, y1 - y0
        i = (ix - x0) * wy + (iy - y0)
        m = wx * wy

        gz = np.where(isg, z, 0.0)
        gf = isg.astype(float)
        gwv = w * gf
        parts = {
            "n": np.bincount(i, minlength=m).astype(float),
            "ng": np.bincount(i, gf, m),
            "gsum": np.bincount(i, gz, m),
            "gsq": np.bincount(i, gz * gz, m),
            "gw": np.bincount(i, gwv, m),
            "gwz": np.bincount(i, gwv * z, m),
            "gwz2": np.bincount(i, gwv * z * z, m),
        }
        for k, v in parts.items():
            self.a[k].reshape(self.n, self.n)[x0:x1, y0:y1] += v.reshape(wx, wy)

        # max and min have no bincount form; ufunc.at is the scatter version
        # and runs over this sweep's points only, never over the map
        flat = ix * self.n + iy
        np.maximum.at(self.zmax, flat, z)
        np.minimum.at(self.gmin, flat, np.where(isg, z, np.inf))


class DenseMap:
    """The tier stack. Query reads the finest tier that has evidence."""

    def __init__(self, tiers=TIERS, trust=20.0):
        self.tiers = [Tier(r, h) for r, h in tiers]
        self.trust = trust
        self.dz = 0.0

    def _align_z(self, x, y, z, isg):
        """Robust z offset taking this sweep onto the map it overlaps.

        NOT optional, and leaving it out is what a fast rewrite quietly loses.
        Measured on seq 00, the same patch of road disagrees between sweeps by
        a median 7.1 cm, ramping about 24 cm over 20 m of travel -- a pitch
        error in the pose chain, not noise. Accumulated raw it lands as
        apparent roughness: the first version of this class omitted the
        correction and gave 51.0% drivable on ground-truth road where the
        sparse map gave 64.2%, a 13 pp loss that looked like a resolution
        artefact and was not.

        Estimated on the coarsest tier, which has the most cells carrying
        enough evidence to compare against.
        """
        t = self.tiers[-1]
        if not isg.any() or t.a["ng"].sum() == 0:
            return 0.0
        # A median needs a sample, not the census. Indexing every ground point
        # cost 62 ms a frame to refine an estimate that is already stable on a
        # few thousand; the stride is deterministic so the result is
        # reproducible run to run.
        gx, gy, gz = x[isg], y[isg], z[isg]
        step = max(1, len(gx) // 12000)
        gx, gy, gz = gx[::step], gy[::step], gz[::step]
        idx, ok = t.index(gx, gy)
        if not ok.any():
            return 0.0
        i, zi = idx[ok], gz[ok]
        ng = t.a["ng"][i]
        seen = ng >= 3
        if seen.sum() < 100:
            return 0.0
        have = t.a["gsum"][i][seen] / ng[seen]
        dz = float(np.median(have - zi[seen]))
        return dz if abs(dz) <= 0.5 else 0.0

    def ingest(self, pts_velo, lab, T_w_velo, moving=None, groundcls=(0, 1)):
        if moving is not None:
            keep = ~moving
            pts_velo, lab = pts_velo[keep], lab[keep]
        rng = np.linalg.norm(pts_velo[:, :2], axis=1)
        near = rng <= self.trust
        p, l, rng = pts_velo[near], lab[near], rng[near]
        if not len(p):
            return self

        w = 1.0 / (SIGMA0 ** 2 + (SIGMA_R * rng) ** 2)
        wpt = p @ T_w_velo[:3, :3].T + T_w_velo[:3, 3]
        isg = np.isin(l, groundcls)

        pos = T_w_velo[:2, 3]
        for t in self.tiers:
            t.move(pos)

        self.dz = self._align_z(wpt[:, 0], wpt[:, 1], wpt[:, 2], isg)
        zc = wpt[:, 2] + self.dz
        for t in self.tiers:
            t.add(wpt[:, 0], wpt[:, 1], zc, isg, w)
        return self

    def cells(self, level=0, min_pts=1):
        """One tier as the dict shape grid25/terrain_cells expect."""
        t = self.tiers[level]
        ok = t.a["n"] >= min_pts
        cx, cy = t.centres()
        out = {k: v[ok] for k, v in t.a.items()}
        out["zmax"] = t.zmax[ok]
        out["gmin"] = t.gmin[ok]
        out["ix"] = np.floor(cx[ok] / t.res).astype(np.int64)
        out["iy"] = np.floor(cy[ok] / t.res).astype(np.int64)
        out["zmin"] = out["zomin"] = out["gmin"]
        out["zsum"] = out["gsum"]        # no separate all-point statistics
        out["zsq"] = out["gsq"]
        out["hist"] = np.zeros((len(sel), 8))
        return out

    def stats(self):
        return {"tiers": len(self.tiers),
                "cells_allocated": sum(t.n * t.n for t in self.tiers),
                "cells_occupied": int(sum((t.a["n"] > 0).sum() for t in self.tiers))}


def selftest():
    rng = np.random.default_rng(0)
    T = np.eye(4)

    m = DenseMap(tiers=((0.5, 10.0),), trust=20.0)
    p = rng.uniform(-8, 8, (5000, 3))
    p[:, 2] = 1.0
    lab = np.zeros(len(p), int)
    m.ingest(p, lab, T)
    t = m.tiers[0]
    assert t.a["n"].sum() == len(p), (t.a["n"].sum(), len(p))
    occ = t.a["n"] > 0
    # every point here is labelled ground, so the ground mean is the height
    assert np.allclose(t.a["gsum"][occ] / t.a["ng"][occ], 1.0)
    assert np.array_equal(t.a["ng"], t.a["n"])

    # a point must land in the cell whose centre is nearest it
    i, ok = t.index(np.array([2.26]), np.array([-3.71]))
    cx, cy = t.centres()
    assert abs(cx[i[0]] - 2.26) <= t.res / 2 + 1e-9
    assert abs(cy[i[0]] + 3.71) <= t.res / 2 + 1e-9

    # move keeps the overlap and blanks only the new strip
    before = t.a["n"].reshape(t.n, t.n).copy()
    m2 = DenseMap(tiers=((0.5, 10.0),))
    m2.ingest(p, lab, T)
    T2 = np.eye(4); T2[0, 3] = 2.0            # 4 cells at 0.5 m
    m2.tiers[0].move(T2[:2, 3])
    after = m2.tiers[0].a["n"].reshape(t.n, t.n)
    assert (after[-4:, :] == 0).all(), "new strip not cleared"
    assert np.array_equal(after[:-4, :], before[4:, :]), "overlap not preserved"

    # accumulating twice must double the counts, exactly
    m3 = DenseMap(tiers=((0.5, 10.0),))
    m3.ingest(p, lab, T); one = m3.tiers[0].a["n"].copy()
    m3.ingest(p, lab, T)
    assert np.array_equal(m3.tiers[0].a["n"], 2 * one)

    # a point beyond a tier's extent must be dropped, not wrapped
    far = np.array([[500.0, 0.0, 0.0]])
    m4 = DenseMap(tiers=((0.5, 10.0),), trust=1e9)
    m4.ingest(far, np.zeros(1, int), T)
    assert m4.tiers[0].a["n"].sum() == 0, "out-of-extent point was not dropped"

    # the windowed bincount must give exactly what a full-tier bincount would
    m5 = DenseMap(tiers=((0.5, 10.0),))
    q = rng.uniform(-9, 9, (4000, 3)); q[:, 2] = rng.normal(0, 1, 4000)
    m5.ingest(q, np.zeros(len(q), int), T)
    t5 = m5.tiers[0]
    ix = np.floor(q[:, 0] / t5.res).astype(np.int64) + t5.n // 2
    iy = np.floor(q[:, 1] / t5.res).astype(np.int64) + t5.n // 2
    ok = (ix >= 0) & (ix < t5.n) & (iy >= 0) & (iy < t5.n)
    ref = np.bincount((ix[ok] * t5.n + iy[ok]), q[ok, 2], t5.n * t5.n)
    assert np.allclose(t5.a["gsum"], ref), "windowed bincount != full bincount"

    print("dense25 selftest ok")


if __name__ == "__main__":
    selftest()


# --------------------------------------------------------------------------
# numba-backed map. Same semantics as DenseMap above, one fused kernel.
# --------------------------------------------------------------------------

from dense_numba import (scatter, align_offset, shift_tier, NROW,
                         scatter_raw, align_raw,
                         scatter_ring, align_ring, clear_strip,
                         N as R_N, NG as R_NG, GSUM as R_GSUM, GSQ as R_GSQ,
                         GW as R_GW, GWZ as R_GWZ, GWZ2 as R_GWZ2,
                         ZMAX as R_ZMAX, GMIN as R_GMIN)

_ROW = {"n": R_N, "ng": R_NG, "gsum": R_GSUM, "gsq": R_GSQ,
        "gw": R_GW, "gwz": R_GWZ, "gwz2": R_GWZ2}


class FastMap:
    """DenseMap with the scatter, the alignment and the shift in numba.

    The layout is the difference: every tier lives in one flat (9, total)
    buffer with an offset table, so the kernel takes plain arrays and the
    parallel loop can run over tiers without touching shared cells.
    """

    def __init__(self, tiers=TIERS, trust=20.0):
        self.res = np.array([t[0] for t in tiers], float)
        self.n = np.array([int(round(2 * h / r)) for r, h in tiers], np.int64)
        self.org = np.zeros((len(tiers), 2))
        sizes = self.n * self.n
        self.off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.acc = np.zeros((NROW, int(sizes.sum())))
        self.acc[R_ZMAX] = -np.inf
        self.acc[R_GMIN] = np.inf
        self.start = np.zeros((len(tiers), 2), np.int64)
        self.trust = float(trust)
        self.dz = 0.0

    def _move(self, pos):
        """Recentre by advancing the ring origin, not by moving the data.

        The physical version relocated 980k cells across nine rows every frame
        -- about 40 ms, five times the scatter it served. This clears only the
        strip that came into view, which at 5 cm and 0.9 m of travel is 18 rows
        of 480 instead of all 480.
        """
        for t in range(len(self.res)):
            d = np.rint((pos - self.org[t]) / self.res[t]).astype(np.int64)
            if not d.any():
                continue
            nt = int(self.n[t])
            dx, dy = int(d[0]), int(d[1])
            if abs(dx) >= nt or abs(dy) >= nt:
                # travelled further than the tier is wide: nothing survives
                a, b = self.off[t], self.off[t + 1]
                self.acc[:, a:b] = 0.0
                self.acc[R_ZMAX, a:b] = -np.inf
                self.acc[R_GMIN, a:b] = np.inf
            else:
                # rows leaving the trailing edge become the new leading edge
                sx0 = int(self.start[t, 0]) if dx > 0 else                     int(self.start[t, 0] + nt + dx)
                sy0 = int(self.start[t, 1]) if dy > 0 else                     int(self.start[t, 1] + nt + dy)
                clear_strip(self.acc, self.off, self.n, t,
                            sx0 % nt, abs(dx), sy0 % nt, abs(dy))
            self.start[t, 0] = (self.start[t, 0] + dx) % nt
            self.start[t, 1] = (self.start[t, 1] + dy) % nt
            self.org[t] = self.org[t] + d * self.res[t]

    def ingest(self, pts_velo, lab, T_w_velo, moving=None, groundcls=(0, 1)):
        """Fold one sweep in. The kernels read the raw points directly."""
        if moving is not None:
            keep = ~moving
            pts_velo, lab = pts_velo[keep], lab[keep]
        if not len(pts_velo):
            return self

        # No copies here. pts_velo[:, :3] is a non-contiguous view, so
        # ascontiguousarray duplicated 2.9 MB every frame purely to satisfy the
        # kernel signature; the kernel now indexes columns itself and takes the
        # sweep as it arrives, extra columns and all. asarray likewise returns
        # the same buffer when the dtype already matches, where astype always
        # copies.
        pts = pts_velo if pts_velo.flags["C_CONTIGUOUS"] else             np.ascontiguousarray(pts_velo)
        lab = np.asarray(lab, np.int64)
        # a lookup indexed by label beats np.isin inside a loop, and lets the
        # kernel decide ground-ness without a precomputed mask array
        gnd = np.zeros(int(max(lab.max(), max(groundcls))) + 1, np.bool_)
        for c in groundcls:
            gnd[c] = True
        R = np.ascontiguousarray(T_w_velo[:3, :3])
        tv = np.ascontiguousarray(T_w_velo[:3, 3])
        t2 = self.trust * self.trust

        self._move(T_w_velo[:2, 3])

        # drift compensation, on the coarsest tier -- the same correction the
        # sparse map applies, and the thing whose absence cost 13 pp of
        # drivable road when this class was first written without it
        last = len(self.res) - 1
        diffs = align_ring(pts, lab, R, tv, t2, gnd, self.org, self.res,
                           self.n, self.off, self.start, self.acc, last)
        dz = float(np.median(diffs)) if len(diffs) >= 100 else 0.0
        self.dz = dz if abs(dz) <= 0.5 else 0.0

        scatter_ring(pts, lab, R, tv, t2, gnd, SIGMA0 ** 2, SIGMA_R ** 2,
                     self.dz, self.org, self.res, self.n, self.off,
                     self.start, self.acc)
        return self

    def cells(self, level=0, min_pts=1):
        a, b = self.off[level], self.off[level + 1]
        nt, r = int(self.n[level]), self.res[level]
        sx, sy = int(self.start[level, 0]), int(self.start[level, 1])
        blk = self.acc[:, a:b]
        ok = blk[R_N] >= min_pts
        # Undo the ring rotation in the INDEX, not in the data. Rolling the
        # buffer straight moved 9 x 250k cells on every query to read a few
        # thousand of them, and cost more than the drivability it fed. Here
        # only the occupied cells are converted, and the conversion is two
        # modulos.
        sel = np.flatnonzero(ok)
        bi, bj = sel // nt, sel % nt
        mi = (bi - sx) % nt - nt // 2
        mj = (bj - sy) % nt - nt // 2
        out = {k: blk[v][sel] for k, v in _ROW.items()}
        out["zmax"] = blk[R_ZMAX][sel]
        out["gmin"] = blk[R_GMIN][sel]
        out["zmin"] = out["zomin"] = out["gmin"]
        out["zsum"], out["zsq"] = out["gsum"], out["gsq"]
        out["ix"] = mi + int(round(self.org[level, 0] / r))
        out["iy"] = mj + int(round(self.org[level, 1] / r))
        out["hist"] = np.zeros((len(sel), 8))
        return out

    def _grid(self, level, nt):
        if not hasattr(self, "_gcache"):
            self._gcache = {}
        if level not in self._gcache:
            gi = np.arange(nt) - nt // 2
            gx, gy = np.meshgrid(gi, gi, indexing="ij")
            self._gcache[level] = (gx.ravel(), gy.ravel())
        return self._gcache[level]

    def stats(self):
        return {"tiers": len(self.res),
                "cells_allocated": int(self.acc.shape[1]),
                "cells_occupied": int((self.acc[R_N] > 0).sum())}
