"""
Phase 3 - export real geometry for the mechanism demos.

Pulls two things out of the KITTI scan:
  1. one isolated object (a car / pole sized cluster) for the PointNet demos
  2. a top-down slice of the scene for the grid / voxel demos

Run:
    .venv\\Scripts\\python.exe scripts\\05_export_phase3.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web"
OUT.mkdir(exist_ok=True)

scans = sorted((ROOT / "data" / "raw").rglob("velodyne_points/data/*.bin"))
if not scans:
    sys.exit("No scans found under data/raw.")

d = np.fromfile(scans[0], dtype=np.float32).reshape(-1, 4)
x, y, z = d[:, 0], d[:, 1], d[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)

# ground plane, recovered the same way as script 04
hist, edges = np.histogram(z[r < 12], bins=400, range=(-3, 1))
ground_z = edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2
print(f"ground plane at z = {ground_z:.3f} m")

# ------------------------------------------------------------------ #
# 1. find one clean, isolated object
# ------------------------------------------------------------------ #
import open3d as o3d

above = (z > ground_z + 0.25) & (r < 30)
pts = d[above, :3].astype(np.float64)
print(f"non-ground points within 30 m: {len(pts):,}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
labels = np.array(pcd.cluster_dbscan(eps=0.45, min_points=15, print_progress=False))
print(f"clusters found: {labels.max() + 1}")

best, best_score = None, -1e9
for k in range(labels.max() + 1):
    m = labels == k
    n = int(m.sum())
    if n < 300 or n > 1600:
        continue
    p = pts[m]
    ext = p.max(0) - p.min(0)
    # want something car-ish: 2-6 m long, 1-3 m wide, 0.8-2.5 m tall
    if not (2.5 < max(ext[0], ext[1]) < 5.0):
        continue
    if not (0.9 < min(ext[0], ext[1]) < 2.6):
        continue
    if not (0.6 < ext[2] < 2.2):
        continue
    dist = np.linalg.norm(p.mean(0)[:2])
    score = n - 60 * dist          # strongly prefer dense and close
    if score > best_score:
        best_score, best = score, (k, n, ext, p, dist)

if best is None:
    sys.exit("No suitable cluster found - loosen the filters.")

k, n, ext, obj, dist = best
print(f"\nchose cluster #{k}")
print(f"  points   {n}")
print(f"  extent   {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")
print(f"  distance {dist:.2f} m from sensor")

obj_c = obj - obj.mean(0)          # centre it for display
scale = float(np.abs(obj_c).max())

# ------------------------------------------------------------------ #
# 2. top-down slice for the grid demos
# ------------------------------------------------------------------ #
RADIUS = 24.0
sl = (r < RADIUS) & (z > ground_z + 0.10)
bev = d[sl, :2]
if len(bev) > 9000:
    idx = np.random.default_rng(0).choice(len(bev), 9000, replace=False)
    bev = bev[idx]
print(f"\nBEV slice: {len(bev):,} non-ground points inside {RADIUS:.0f} m")

# ------------------------------------------------------------------ #
# 3. real points-per-cell comparison: square vs polar
# ------------------------------------------------------------------ #
full = d[r < RADIUS, :2]
fx, fy = full[:, 0], full[:, 1]
frad = np.sqrt(fx**2 + fy**2)
fth = np.arctan2(fy, fx)


def occupancy_spread(cells):
    """cells: integer cell id per point -> stats on points per occupied cell"""
    _, counts = np.unique(cells, return_counts=True)
    return {
        "cells": int(len(counts)),
        "mean": round(float(counts.mean()), 2),
        "median": int(np.median(counts)),
        "p95": int(np.percentile(counts, 95)),
        "max": int(counts.max()),
        "hist": np.histogram(np.clip(counts, 0, 60), bins=20, range=(0, 60))[0].tolist(),
    }


SQ = 0.4                                   # 40 cm square cells
sq_id = (np.floor(fx / SQ).astype(np.int64) * 100000
         + np.floor(fy / SQ).astype(np.int64))

NR, NTH = 60, 360                          # 60 radial rings, 360 angular sectors
pr_id = (np.clip((frad / RADIUS * NR).astype(np.int64), 0, NR - 1) * 100000
         + np.clip(((fth + np.pi) / (2 * np.pi) * NTH).astype(np.int64), 0, NTH - 1))

cart = occupancy_spread(sq_id)
poly = occupancy_spread(pr_id)
print(f"\npoints per occupied cell, {len(full):,} points inside {RADIUS:.0f} m")
print(f"  cartesian {SQ*100:.0f} cm : {cart['cells']:>6,} cells  "
      f"median {cart['median']:>3}  p95 {cart['p95']:>4}  max {cart['max']:>5}")
print(f"  polar {NR}x{NTH}      : {poly['cells']:>6,} cells  "
      f"median {poly['median']:>3}  p95 {poly['p95']:>4}  max {poly['max']:>5}")

# ------------------------------------------------------------------ #
payload = {
    "object": {
        "n": int(n),
        "extent": [round(float(v), 2) for v in ext],
        "distance": round(float(dist), 2),
        "scale": round(scale, 3),
        "pts": [[round(float(a), 3) for a in p] for p in obj_c],
    },
    "bev": {
        "radius": RADIUS,
        "n": int(len(bev)),
        "pts": [[round(float(a), 2) for a in p] for p in bev],
    },
    "cellstats": {"sq": SQ, "nr": NR, "nth": NTH,
                  "cart": cart, "polar": poly,
                  "total": int(len(full))},
    "groundZ": round(float(ground_z), 3),
}
p = OUT / "phase3.json"
p.write_text(json.dumps(payload))
print(f"\nwrote {p.relative_to(ROOT)}  {p.stat().st_size/1e3:.0f} kB")
