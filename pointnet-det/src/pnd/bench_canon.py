"""
Is PCA actually cheaper than a T-Net?

The claim under test: replacing PointNet's learned input transform with PCA
moves work off the GPU and is faster overall. That is a systems claim, so it
gets measured rather than argued.

What is compared, on the workload this repo actually runs -- B clusters of P
points each, which is what cluster-then-classify feeds the network:

    tnet3       learned 3x3 transform  (PointNet's input T-Net)
    tnet64      learned 64x64 transform (the feature T-Net we intend to delete)
    eigh_loop   np.linalg.eigh, one call per cluster - the naive way
    eigh_batch  np.linalg.eigh on a stacked (B, 3, 3) array
    analytic    closed-form symmetric 3x3 eigendecomposition, numba, parallel
    analytic2d  closed-form symmetric 2x2 - the yaw-only variant

    python -m pnd.bench_canon
"""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from numba import njit, prange

# --------------------------------------------------------------------------- #
# closed-form symmetric eigendecomposition
# --------------------------------------------------------------------------- #
# A symmetric 3x3 has analytic eigenvalues via the trigonometric solution to its
# characteristic cubic. No iteration, no LAPACK call, no per-cluster Python
# overhead -- which is what actually dominates at these sizes.


@njit(cache=True, fastmath=True)
def _eig3_sym(a00, a01, a02, a11, a12, a22):
    """Eigenvalues of a symmetric 3x3, descending. Smith's trigonometric method."""
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3.0
    if p1 <= 1e-18:
        v = np.empty(3, np.float64)
        v[0] = a00
        v[1] = a11
        v[2] = a22
        # descending
        for i in range(3):
            for j in range(i + 1, 3):
                if v[j] > v[i]:
                    t = v[i]
                    v[i] = v[j]
                    v[j] = t
        return v[0], v[1], v[2]

    p2 = ((a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2) + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    ip = 1.0 / p
    b00 = ip * (a00 - q)
    b11 = ip * (a11 - q)
    b22 = ip * (a22 - q)
    b01 = ip * a01
    b02 = ip * a02
    b12 = ip * a12
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = detB / 2.0
    if r <= -1.0:
        phi = np.pi / 3.0
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = np.arccos(r) / 3.0
    e0 = q + 2.0 * p * np.cos(phi)
    e2 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e1 = 3.0 * q - e0 - e2
    return e0, e1, e2


@njit(parallel=True, cache=True, fastmath=True)
def pca3_batch(pts, offs, out_R, out_t, out_lam):
    """Batched 3D PCA over a ragged set of clusters.

    pts   (N, 3)  all points, clusters laid out contiguously
    offs  (B+1,)  cluster b occupies pts[offs[b]:offs[b+1]]
    """
    B = offs.shape[0] - 1
    for b in prange(B):
        lo, hi = offs[b], offs[b + 1]
        n = hi - lo
        if n < 3:
            for i in range(3):
                for j in range(3):
                    out_R[b, i, j] = 1.0 if i == j else 0.0
            continue

        cx = 0.0; cy = 0.0; cz = 0.0
        for i in range(lo, hi):
            cx += pts[i, 0]; cy += pts[i, 1]; cz += pts[i, 2]
        inv = 1.0 / n
        cx *= inv; cy *= inv; cz *= inv
        out_t[b, 0] = cx; out_t[b, 1] = cy; out_t[b, 2] = cz

        c00 = 0.0; c01 = 0.0; c02 = 0.0; c11 = 0.0; c12 = 0.0; c22 = 0.0
        for i in range(lo, hi):
            dx = pts[i, 0] - cx; dy = pts[i, 1] - cy; dz = pts[i, 2] - cz
            c00 += dx * dx; c01 += dx * dy; c02 += dx * dz
            c11 += dy * dy; c12 += dy * dz; c22 += dz * dz
        d = 1.0 / max(n - 1, 1)
        c00 *= d; c01 *= d; c02 *= d; c11 *= d; c12 *= d; c22 *= d

        e0, e1, e2 = _eig3_sym(c00, c01, c02, c11, c12, c22)
        out_lam[b, 0] = e0; out_lam[b, 1] = e1; out_lam[b, 2] = e2

        # eigenvector per eigenvalue from the cross product of two rows of
        # (C - lambda I); pick the pair with the largest cross product norm
        for k in range(3):
            lam = e0 if k == 0 else (e1 if k == 1 else e2)
            m00 = c00 - lam; m11 = c11 - lam; m22 = c22 - lam
            r0x, r0y, r0z = m00, c01, c02
            r1x, r1y, r1z = c01, m11, c12
            r2x, r2y, r2z = c02, c12, m22
            ax = r0y * r1z - r0z * r1y
            ay = r0z * r1x - r0x * r1z
            az = r0x * r1y - r0y * r1x
            na = ax * ax + ay * ay + az * az
            bx = r0y * r2z - r0z * r2y
            by = r0z * r2x - r0x * r2z
            bz = r0x * r2y - r0y * r2x
            nb = bx * bx + by * by + bz * bz
            gx = r1y * r2z - r1z * r2y
            gy = r1z * r2x - r1x * r2z
            gz = r1x * r2y - r1y * r2x
            ng = gx * gx + gy * gy + gz * gz
            if na >= nb and na >= ng:
                vx, vy, vz, nn = ax, ay, az, na
            elif nb >= ng:
                vx, vy, vz, nn = bx, by, bz, nb
            else:
                vx, vy, vz, nn = gx, gy, gz, ng
            if nn < 1e-20:
                vx = 1.0 if k == 0 else 0.0
                vy = 1.0 if k == 1 else 0.0
                vz = 1.0 if k == 2 else 0.0
                nn = 1.0
            s = 1.0 / np.sqrt(nn)
            vx *= s; vy *= s; vz *= s

            # sign from the third moment along this axis
            m3 = 0.0
            for i in range(lo, hi):
                t = ((pts[i, 0] - cx) * vx + (pts[i, 1] - cy) * vy
                     + (pts[i, 2] - cz) * vz)
                m3 += t * t * t
            if m3 < 0.0:
                vx = -vx; vy = -vy; vz = -vz
            out_R[b, k, 0] = vx; out_R[b, k, 1] = vy; out_R[b, k, 2] = vz

        det = (out_R[b, 0, 0] * (out_R[b, 1, 1] * out_R[b, 2, 2] - out_R[b, 1, 2] * out_R[b, 2, 1])
               - out_R[b, 0, 1] * (out_R[b, 1, 0] * out_R[b, 2, 2] - out_R[b, 1, 2] * out_R[b, 2, 0])
               + out_R[b, 0, 2] * (out_R[b, 1, 0] * out_R[b, 2, 1] - out_R[b, 1, 1] * out_R[b, 2, 0]))
        if det < 0.0:
            out_R[b, 2, 0] = -out_R[b, 2, 0]
            out_R[b, 2, 1] = -out_R[b, 2, 1]
            out_R[b, 2, 2] = -out_R[b, 2, 2]


@njit(parallel=True, cache=True, fastmath=True)
def pca2_batch(pts, offs, out_yaw):
    """Yaw-only: closed-form 2x2 symmetric eigen on the (x, y) footprint."""
    B = offs.shape[0] - 1
    for b in prange(B):
        lo, hi = offs[b], offs[b + 1]
        n = hi - lo
        if n < 3:
            out_yaw[b] = 0.0
            continue
        cx = 0.0; cy = 0.0
        for i in range(lo, hi):
            cx += pts[i, 0]; cy += pts[i, 1]
        cx /= n; cy /= n
        a = 0.0; bb = 0.0; c = 0.0
        for i in range(lo, hi):
            dx = pts[i, 0] - cx; dy = pts[i, 1] - cy
            a += dx * dx; bb += dx * dy; c += dy * dy
        # dominant eigenvector of [[a, bb], [bb, c]], closed form
        tr = a + c
        det = a * c - bb * bb
        disc = np.sqrt(max(tr * tr / 4.0 - det, 0.0))
        lam = tr / 2.0 + disc
        if abs(bb) > 1e-15:
            vx, vy = lam - c, bb
        else:
            vx, vy = (1.0, 0.0) if a >= c else (0.0, 1.0)
        nn = np.sqrt(vx * vx + vy * vy)
        vx /= nn; vy /= nn
        m3 = 0.0
        for i in range(lo, hi):
            t = (pts[i, 0] - cx) * vx + (pts[i, 1] - cy) * vy
            m3 += t * t * t
        if m3 < 0.0:
            vx = -vx; vy = -vy
        out_yaw[b] = np.arctan2(vy, vx)


# --------------------------------------------------------------------------- #
# T-Net, for the other side of the comparison
# --------------------------------------------------------------------------- #
if HAVE_TORCH:
    class TNet(nn.Module):
        """PointNet's transform network, verbatim in shape."""

        def __init__(self, k: int):
            super().__init__()
            self.k = k
            self.conv = nn.Sequential(
                nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, k * k),
            )

        def forward(self, x):                       # (B, k, P)
            f = self.conv(x).max(dim=2).values
            m = self.fc(f).view(-1, self.k, self.k)
            return m + torch.eye(self.k, device=x.device)

        def n_params(self):
            return sum(p.numel() for p in self.parameters())


def timeit(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    B, P = 64, 256          # clusters per frame, points per cluster
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(B * P, 3)).astype(np.float64)
    offs = (np.arange(B + 1) * P).astype(np.int64)

    print(f"workload: {B} clusters x {P} points  "
          f"({B*P:,} points, one frame's worth)\n")

    R = np.zeros((B, 3, 3)); t = np.zeros((B, 3)); lam = np.zeros((B, 3))
    yaw = np.zeros(B)
    pca3_batch(pts, offs, R, t, lam)          # jit warm-up
    pca2_batch(pts, offs, yaw)

    stacked = pts.reshape(B, P, 3)
    covs = np.einsum("bpi,bpj->bij", stacked - stacked.mean(1, keepdims=True),
                     stacked - stacked.mean(1, keepdims=True)) / (P - 1)

    rows = []
    rows.append(("eigh_loop  (numpy, per cluster)",
                 timeit(lambda: [np.linalg.eigh(covs[i]) for i in range(B)]), "CPU"))
    rows.append(("eigh_batch (numpy, stacked)",
                 timeit(lambda: np.linalg.eigh(covs)), "CPU"))
    rows.append(("analytic3  (numba, parallel)",
                 timeit(lambda: pca3_batch(pts, offs, R, t, lam)), "CPU"))
    rows.append(("analytic2d (numba, parallel)",
                 timeit(lambda: pca2_batch(pts, offs, yaw)), "CPU"))

    if HAVE_TORCH:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x3 = torch.randn(B, 3, P, device=dev)
        x64 = torch.randn(B, 64, P, device=dev)
        t3 = TNet(3).to(dev).eval()
        t64 = TNet(64).to(dev).eval()

        def run(net, x):
            def f():
                with torch.no_grad():
                    net(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
            return f

        rows.append((f"tnet3      ({t3.n_params()/1e3:.0f}k params)",
                     timeit(run(t3, x3)), dev.upper()))
        rows.append((f"tnet64     ({t64.n_params()/1e3:.0f}k params)",
                     timeit(run(t64, x64)), dev.upper()))

    print(f"{'method':<36}{'device':>8}{'ms/frame':>12}{'vs fastest':>12}")
    print("-" * 68)
    best = min(r[1] for r in rows)
    for name, ms, dev in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<36}{dev:>8}{ms:>12.4f}{ms/best:>11.1f}x")

    print()
    print("Correctness: analytic vs LAPACK eigenvalues")
    ev_ref = np.sort(np.linalg.eigvalsh(covs), axis=1)[:, ::-1]
    err = np.abs(np.sort(lam, axis=1)[:, ::-1] - ev_ref).max()
    print(f"  max eigenvalue error {err:.3e}   {'OK' if err < 1e-9 else 'FAIL'}")


if __name__ == "__main__":
    main()
