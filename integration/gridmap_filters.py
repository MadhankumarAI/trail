"""grid_map's terrain filter chain, on our 2.5D layer.

This is grid_map used to BUILD the map, not merely to publish it. The chain is
the one in grid_map_demos/config/filters_demo_filter_chain.yaml, transcribed
from grid_map_filters/src/NormalVectorsFilter.cpp:

    elevation_smooth  MeanInRadiusFilter(elevation, radius)
    normal_vectors_*  NormalVectorsFilter(elevation, radius)   <- PCA plane
    slope             acos(normal_vectors_z)                   [radians]
    roughness         abs(elevation - elevation_smooth)
    traversability    0.5*(1 - slope/0.6) + 0.5*(1 - roughness/0.1), clamped

WHY THIS IS DIFFERENT FROM WHAT WE HAD
--------------------------------------
Our own terrain_cells computes

    slope      atan(largest neighbouring height difference / cell size)
    roughness  tilt-corrected standard deviation of heights inside one cell

Both are defensible, and both are cruder than grid_map's in a specific way:

  - our slope is a MAX over four neighbours, so a single noisy cell tilts it,
    and it can only see one cell out. grid_map fits a plane to everything in a
    radius and takes the normal, which is the standard estimator and averages
    that noise away.

  - our roughness lives INSIDE a cell, so it is blind to bumpiness at any scale
    larger than one cell. grid_map measures deviation from the local mean
    surface, so a cell that is smooth in itself but sits proud of its
    neighbours reads rough, which is what a wheel feels.

NORMAL ESTIMATION, exactly as NormalVectorsFilter::areaSingleNormalComputation
------------------------------------------------------------------------------
    gather every valid (x, y, elevation) within estimationRadius (a CIRCLE,
        not a square window)
    if fewer than 3 points            -> normal = +z
    cov = sumSquared/n - mean*mean^T
    eigenvalues ascending; if eigenvalues[1] <= 1e-8 the data is on a line and
        the normal is undefined  -> normal = +z
    else normal = eigenvector of the SMALLEST eigenvalue
    flip so it points along +z

The degeneracy guard is on the SECOND eigenvalue, not the first. That is the
subtle part: a rank-1 neighbourhood (all points collinear) still has a smallest
eigenvector, and it is meaningless. Dropping that check yields normals that
look fine and point anywhere along ridges and map edges.

ON THE RADIUS
-------------
The demo uses 0.05 m because its map is centimetre-scale; grid_map itself warns
when the radius is below half the resolution. At our 0.4 m cells that would be
a single cell and no plane at all, so the radius here is set in metres from
what a vehicle actually spans, defaulting to 1.0 m -- about a wheelbase's worth
of ground, which is the scale at which slope matters for driving.
"""

import numpy as np

# grid_map demo chain limits, used verbatim for the traversability expression
DEMO_SLOPE_LIM = 0.6        # radians, about 34 degrees
DEMO_ROUGH_LIM = 0.1        # metres


def _disc(radius, res):
    """Cell offsets whose CENTRES lie within `radius` -- grid_map's
    CircleIterator tests the cell centre, not overlap."""
    k = int(np.floor(radius / res))
    off = []
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if (dr * res) ** 2 + (dc * res) ** 2 <= radius * radius:
                off.append((dr, dc))
    return off


def _shift(a, dr, dc, fill):
    """Translate an array, filling what scrolls in -- the map has edges and
    they must not wrap round."""
    out = np.full_like(a, fill)
    r0, r1 = max(dr, 0), a.shape[0] + min(dr, 0)
    c0, c1 = max(dc, 0), a.shape[1] + min(dc, 0)
    out[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def mean_in_radius(elev, res, radius):
    """MeanInRadiusFilter: mean of the valid cells within a radius."""
    valid = np.isfinite(elev)
    z = np.where(valid, elev, 0.0)
    s = np.zeros_like(z, float)
    n = np.zeros_like(z, float)
    for dr, dc in _disc(radius, res):
        s += _shift(z, dr, dc, 0.0)
        n += _shift(valid.astype(float), dr, dc, 0.0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def normal_vectors(elev, res, radius, x0=0.0, y0=0.0):
    """NormalVectorsFilter, area method. Returns (nx, ny, nz).

    Vectorised over the whole grid: the ten windowed sums are accumulated with
    shifts, then every cell's 3x3 covariance is diagonalised in one batched
    eigh. That is the same decomposition Eigen's computeDirect performs, and
    numpy also returns eigenvalues in ascending order, so `col(0)` is column 0
    here too.
    """
    ny_, nx_ = elev.shape
    rows = np.arange(ny_)[:, None] * np.ones((1, nx_))
    cols = np.ones((ny_, 1)) * np.arange(nx_)[None, :]
    # positions only need to be consistent, not absolute: a covariance is
    # invariant to translation, and the normal is what we want
    X = x0 - rows * res
    Y = y0 - cols * res

    valid = np.isfinite(elev)
    Z = np.where(valid, elev, 0.0)
    v = valid.astype(float)

    n = np.zeros_like(Z)
    sx = np.zeros_like(Z); sy = np.zeros_like(Z); sz = np.zeros_like(Z)
    sxx = np.zeros_like(Z); sxy = np.zeros_like(Z); sxz = np.zeros_like(Z)
    syy = np.zeros_like(Z); syz = np.zeros_like(Z); szz = np.zeros_like(Z)
    for dr, dc in _disc(radius, res):
        m = _shift(v, dr, dc, 0.0)
        px = _shift(X * v, dr, dc, 0.0)
        py = _shift(Y * v, dr, dc, 0.0)
        pz = _shift(Z * v, dr, dc, 0.0)
        n += m; sx += px; sy += py; sz += pz
        sxx += _shift(X * X * v, dr, dc, 0.0)
        sxy += _shift(X * Y * v, dr, dc, 0.0)
        sxz += _shift(X * Z * v, dr, dc, 0.0)
        syy += _shift(Y * Y * v, dr, dc, 0.0)
        syz += _shift(Y * Z * v, dr, dc, 0.0)
        szz += _shift(Z * Z * v, dr, dc, 0.0)

    nn = np.maximum(n, 1.0)
    mx, my, mz = sx / nn, sy / nn, sz / nn
    cxx = sxx / nn - mx * mx
    cxy = sxy / nn - mx * my
    cxz = sxz / nn - mx * mz
    cyy = syy / nn - my * my
    cyz = syz / nn - my * mz
    czz = szz / nn - mz * mz

    C = np.empty(elev.shape + (3, 3))
    C[..., 0, 0] = cxx; C[..., 0, 1] = cxy; C[..., 0, 2] = cxz
    C[..., 1, 0] = cxy; C[..., 1, 1] = cyy; C[..., 1, 2] = cyz
    C[..., 2, 0] = cxz; C[..., 2, 1] = cyz; C[..., 2, 2] = czz
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    w, V = np.linalg.eigh(C)              # ascending, like Eigen
    nvec = V[..., :, 0]                   # smallest eigenvalue's eigenvector

    # grid_map's two fallbacks, both to +z
    bad = (n < 3) | (w[..., 1] <= 1e-8)
    nvec = np.where(bad[..., None], np.array([0.0, 0.0, 1.0]), nvec)

    # flip towards the positive z axis
    flip = nvec[..., 2] < 0.0
    nvec = np.where(flip[..., None], -nvec, nvec)

    out = np.where(valid[..., None], nvec, np.nan)
    return out[..., 0], out[..., 1], out[..., 2]


def chain(elev, res, normal_radius=1.0, mean_radius=1.0,
          slope_lim=DEMO_SLOPE_LIM, rough_lim=DEMO_ROUGH_LIM):
    """The whole demo chain. Returns a dict of layers, grid_map's names."""
    smooth = mean_in_radius(elev, res, mean_radius)
    nx, ny, nz = normal_vectors(elev, res, normal_radius)
    slope = np.arccos(np.clip(nz, -1.0, 1.0))          # acos(normal_vectors_z)
    rough = np.abs(elev - smooth)                      # |elev - elev_smooth|
    trav = np.clip(0.5 * (1.0 - slope / slope_lim)
                   + 0.5 * (1.0 - rough / rough_lim), 0.0, 1.0)
    return {
        "elevation_smooth": smooth,
        "normal_vectors_x": nx,
        "normal_vectors_y": ny,
        "normal_vectors_z": nz,
        "slope": slope,
        "roughness": rough,
        "traversability": trav,
    }


def selftest():
    res = 0.4

    # a plane tilted by a known angle must come back with that slope
    for deg in (0.0, 5.0, 15.0, 30.0):
        rows = np.arange(40)[:, None] * np.ones((1, 40))
        z = np.tan(np.radians(deg)) * (-rows * res)      # x = -row*res
        out = chain(z, res, normal_radius=1.2, mean_radius=1.2)
        s = np.degrees(out["slope"][8:-8, 8:-8])
        assert abs(np.median(s) - deg) < 0.5, (deg, np.median(s))

    # a flat plane is perfectly smooth
    flat = np.zeros((30, 30))
    out = chain(flat, res)
    assert np.nanmax(out["roughness"][6:-6, 6:-6]) < 1e-9
    assert np.nanmax(np.abs(out["slope"][6:-6, 6:-6])) < 1e-6
    assert np.nanmin(out["traversability"][6:-6, 6:-6]) > 0.999

    # a single raised cell must read rough, and rougher than its neighbours
    bump = np.zeros((30, 30))
    bump[15, 15] = 0.25
    out = chain(bump, res, normal_radius=1.2, mean_radius=1.2)
    assert out["roughness"][15, 15] > out["roughness"][15, 20]
    assert out["roughness"][15, 15] > 0.1

    # the degeneracy guard: a single row of valid cells is collinear, so the
    # normal is undefined and must fall back to +z rather than to noise
    line = np.full((30, 30), np.nan)
    line[15, :] = 0.0
    nx, ny, nz = normal_vectors(line, res, 1.2)
    assert np.allclose(nz[15, 5:-5], 1.0), np.nanmin(nz[15, 5:-5])

    # holes must not wrap around the map edge
    a = np.zeros((20, 20)); a[0, :] = 5.0
    m = mean_in_radius(a, res, 1.2)
    assert m[19, 10] < 0.001, m[19, 10]

    print("gridmap_filters selftest ok")


if __name__ == "__main__":
    selftest()
