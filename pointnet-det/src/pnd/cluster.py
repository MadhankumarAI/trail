"""
Voxel connected-component clustering.

DBSCAN via open3d takes ~1 s on a full KITTI scan, which is 10x the entire
latency budget. This does the same job in a few milliseconds by quantising to a
voxel grid and running union-find over occupied voxels.

The trade against DBSCAN is real and worth stating: voxel CCL cannot separate
two objects closer together than one voxel, so a pedestrian standing against a
wall merges. DBSCAN has the same failure at eps. Neither recovers from it, which
is why `evaluate.py` reports cluster recall separately from classifier accuracy
-- if the clustering loses an object, no network downstream can find it.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]      # path halving
        x = parent[x]
    return x


@njit(cache=True)
def _union(parent, rank, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


@njit(cache=True)
def _ccl(keys, order, nx, ny, labels_out):
    """Union-find over occupied voxels using a sorted key array.

    `keys` are linear voxel ids, `order` sorts them. Neighbour lookup is a
    binary search into the sorted keys, which avoids building a hash map.
    """
    n = keys.shape[0]
    parent = np.arange(n)
    rank = np.zeros(n, np.int32)

    sorted_keys = np.empty(n, np.int64)
    for i in range(n):
        sorted_keys[i] = keys[order[i]]

    for i in range(n):
        k = sorted_keys[i]
        kz = k // (nx * ny)
        rem = k - kz * nx * ny
        ky = rem // nx
        kx = rem - ky * nx
        # only the 13 "forward" neighbours; the reverse pairs are covered when
        # the other voxel is visited
        for dz in range(0, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dz == 0 and (dy < 0 or (dy == 0 and dx <= 0)):
                        continue
                    nxq = kx + dx
                    nyq = ky + dy
                    nzq = kz + dz
                    if nxq < 0 or nxq >= nx or nyq < 0 or nyq >= ny or nzq < 0:
                        continue
                    nk = nzq * nx * ny + nyq * nx + nxq
                    lo, hi = 0, n
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if sorted_keys[mid] < nk:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo < n and sorted_keys[lo] == nk:
                        _union(parent, rank, i, lo)

    for i in range(n):
        labels_out[i] = _find(parent, i)


def cluster_points(pts: np.ndarray,
                   voxel: float = 0.30,
                   min_points: int = 20,
                   max_points: int = 6000):
    """Cluster (N, 3) points. Returns an int label per point, -1 for noise."""
    if len(pts) == 0:
        return np.full(0, -1, np.int64)

    p = pts[:, :3].astype(np.float64)
    lo = p.min(axis=0)
    ijk = np.floor((p - lo) / voxel).astype(np.int64)
    nx = int(ijk[:, 0].max()) + 2
    ny = int(ijk[:, 1].max()) + 2
    keys = ijk[:, 2] * nx * ny + ijk[:, 1] * nx + ijk[:, 0]

    uniq, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(uniq)
    roots = np.empty(len(uniq), np.int64)
    _ccl(uniq, order, nx, ny, roots)

    # roots index into the sorted array; map back to original voxel order
    vox_label = np.empty(len(uniq), np.int64)
    vox_label[order] = roots
    lab = vox_label[inv]

    # compact ids and drop clusters outside the size window.
    # Vectorised: a Python dict lookup per point cost ~25 ms on a real scan.
    uu, counts = np.unique(lab, return_counts=True)
    keep_mask = (counts >= min_points) & (counts <= max_points)
    new_id = np.full(len(uu), -1, np.int64)
    new_id[keep_mask] = np.arange(int(keep_mask.sum()))
    pos = np.searchsorted(uu, lab)
    return new_id[pos]
