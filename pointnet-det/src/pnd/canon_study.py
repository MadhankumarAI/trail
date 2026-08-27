"""
Does deterministic canonicalisation actually work on LiDAR clusters?

Two properties define a canonicaliser. Both are measured here on real clusters,
not asserted:

  TEST 1 - equivariance
      Rotate the input, and the canonical output must not change.
      canon(R @ P) == canon(P). This is the definition. If it fails, the network
      sees a different object every time the same object turns.

  TEST 2 - viewpoint consistency
      A LiDAR never sees a whole object, only the near surface. The same object
      viewed from two positions must still canonicalise to the same frame,
      otherwise the canonicalisation is encoding sensor position rather than
      object shape.

Test 1 is the easy one and PCA should pass it. Test 2 is where the argument is.

    python -m pnd.canon_study --scan <path to a .bin>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .canon import (canon_none, canon_pca2_yaw, canon_pca3_skew, degeneracy,
                    frame_disagreement, simulate_occlusion, yaw_disagreement)

VARIANTS = {
    "none": canon_none,
    "pca3_skew": canon_pca3_skew,
    "pca2_yaw": canon_pca2_yaw,
}


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_random(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Mean nearest-neighbour distance, symmetric. Cheap for small clusters."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def extract_clusters(scan: Path, min_pts=120, max_pts=3000, max_range=30.0):
    """Ground-remove and cluster one scan. Returns a list of (N, 3) arrays."""
    import open3d as o3d

    d = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hist, edges = np.histogram(z[r < 12], bins=400, range=(-3, 1))
    gz = edges[int(np.argmax(hist))]

    m = (z > gz + 0.25) & (r < max_range)
    pts = d[m, :3].astype(np.float64)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    lab = np.array(pc.cluster_dbscan(eps=0.45, min_points=15, print_progress=False))

    out = []
    for k in range(lab.max() + 1):
        q = pts[lab == k]
        if min_pts <= len(q) <= max_pts:
            out.append(q)
    return out


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    clusters = extract_clusters(args.scan)
    print(f"scan     {args.scan.name}")
    print(f"clusters {len(clusters)}  "
          f"(sizes {min(len(c) for c in clusters)}-{max(len(c) for c in clusters)})")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("  CONDITIONING OF THE REAL CLUSTERS")
    print("=" * 74)
    d21 = np.array([degeneracy(c)[0] for c in clusters])
    d32 = np.array([degeneracy(c)[1] for c in clusters])
    print("Eigenvalue ratios. Near 1.0 means the eigenvectors are arbitrary,")
    print("so the canonical frame is decided by noise.")
    print(f"\n  {'ratio':<14}{'median':>9}{'p75':>9}{'p90':>9}{'> 0.85':>10}")
    print("  " + "-" * 51)
    for nm, v in [("lam2 / lam1", d21), ("lam3 / lam2", d32)]:
        print(f"  {nm:<14}{np.median(v):>9.3f}{pct(v,75):>9.3f}"
              f"{pct(v,90):>9.3f}{100*np.mean(v>0.85):>9.0f}%")

    xy = []
    for c in clusters:
        p = c[:, :2] - c[:, :2].mean(0)
        ev = np.linalg.eigvalsh((p.T @ p) / max(len(p) - 1, 1))
        ev = np.sort(np.maximum(ev, 1e-12))[::-1]
        xy.append(ev[1] / ev[0])
    xy = np.array(xy)
    print(f"  {'2D footprint':<14}{np.median(xy):>9.3f}{pct(xy,75):>9.3f}"
          f"{pct(xy,90):>9.3f}{100*np.mean(xy>0.85):>9.0f}%")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("  TEST 1 - EQUIVARIANCE  (complete points, the easy case)")
    print("=" * 74)
    print("Rotate the cluster, canonicalise, compare against canonicalising the")
    print("unrotated cluster. A correct canonicaliser gives ~0 for its own")
    print("nuisance group.\n")
    print(f"  {'variant':<12}{'yaw only':>22}{'full SO(3)':>22}")
    print(f"  {'':<12}{'median':>11}{'p90':>11}{'median':>11}{'p90':>11}")
    print("  " + "-" * 56)

    for name, fn in VARIANTS.items():
        yaw_err, so3_err = [], []
        for c in clusters:
            base, _ = fn(c)
            for _ in range(args.trials):
                Ry = rot_z(rng.uniform(-np.pi, np.pi))
                got, _ = fn((Ry @ c.T).T)
                yaw_err.append(chamfer(base, got))
                Rr = rot_random(rng)
                got2, _ = fn((Rr @ c.T).T)
                so3_err.append(chamfer(base, got2))
        y, s = np.array(yaw_err), np.array(so3_err)
        print(f"  {name:<12}{np.median(y):>11.4f}{pct(y,90):>11.4f}"
              f"{np.median(s):>11.4f}{pct(s,90):>11.4f}")
    print("\n  (chamfer distance in unit-sphere units; 0 = perfectly equivariant)")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("  TEST 2 - VIEWPOINT CONSISTENCY  (partial observation, the real case)")
    print("=" * 74)
    print("Show each cluster to two sensors 90 degrees apart, keeping only the")
    print("near half each time - what a LiDAR actually gets. Then ask whether the")
    print("two canonical frames agree.\n")
    print(f"  {'variant':<12}{'full frame disagree':>24}{'yaw disagree':>20}")
    print(f"  {'':<12}{'median':>12}{'p90':>12}{'median':>10}{'p90':>10}")
    print("  " + "-" * 56)

    for name, fn in VARIANTS.items():
        if name == "none":
            continue
        fd, yd = [], []
        for c in clusters:
            ctr = c.mean(0)
            rad = float(np.linalg.norm(c - ctr, axis=1).max()) * 4 + 5
            for _ in range(args.trials):
                a = rng.uniform(-np.pi, np.pi)
                b = a + np.pi / 2
                sa = ctr + np.array([np.cos(a), np.sin(a), 0.2]) * rad
                sb = ctr + np.array([np.cos(b), np.sin(b), 0.2]) * rad
                pa = simulate_occlusion(c, sa, keep=0.5)
                pb = simulate_occlusion(c, sb, keep=0.5)
                _, fa = fn(pa)
                _, fb = fn(pb)
                fd.append(frame_disagreement(fa.R, fb.R))
                yd.append(yaw_disagreement(fa.R, fb.R))
        fd, yd = np.array(fd), np.array(yd)
        print(f"  {name:<12}{np.median(fd):>12.1f}{pct(fd,90):>12.1f}"
              f"{np.median(yd):>10.1f}{pct(yd,90):>10.1f}")
    print("\n  (degrees. 0 = the two viewpoints agree on the object's frame)")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("  HOW OFTEN IS THE SIGN ACTUALLY DETERMINED?")
    print("=" * 74)
    for name, fn in VARIANTS.items():
        if name == "none":
            continue
        amb = deg = 0
        for c in clusters:
            _, f = fn(c)
            amb += f.skew_ambiguous
            deg += f.degenerate
        n = len(clusters)
        print(f"  {name:<12} skew ambiguous {100*amb/n:5.1f}%   "
              f"ill-conditioned {100*deg/n:5.1f}%")
    print()


if __name__ == "__main__":
    main()
