"""
Phase 2 - export one real KITTI scan (plus its camera image) into a compact
payload the browser explorer can load.

Produces:
    web/payload.json.txt   base64 of a packed binary blob
    web/frame.jpg.txt      base64 JPEG of the matching camera frame

Run:
    .venv\\Scripts\\python.exe scripts\\02_export_web.py
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web"
OUT.mkdir(exist_ok=True)

FRAME = 0
SENSOR_HEIGHT = 1.73


def recover_rings(x, y):
    """KITTI strips the laser id, but it is recoverable.

    Points are stored laser by laser, top beam first. Within one laser the
    head rotates counter-clockwise, so azimuth increases monotonically and
    then wraps once from +pi to -pi. Those wraps are the ring boundaries:
    exactly 64 of them in an HDL-64E sweep.

    The sweep starts part-way through the top laser's rotation, so the very
    first partial segment belongs to the same laser as the one after it.
    Subtracting one and clamping merges them, giving rings 0..63.
    """
    azi = np.arctan2(y, x)
    wrap = np.zeros(len(azi), dtype=np.int32)
    wrap[1:] = (np.diff(azi) < -np.pi).astype(np.int32)
    return np.maximum(np.cumsum(wrap) - 1, 0)


def main():
    scans = sorted((ROOT / "data" / "raw").rglob("velodyne_points/data/*.bin"))
    if not scans:
        sys.exit("No scans found.")
    path = scans[FRAME]
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    x, y, z, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    rng = np.sqrt(x**2 + y**2 + z**2)
    azi = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arcsin(np.clip(z / np.maximum(rng, 1e-9), -1, 1)))
    ring = recover_rings(x, y)

    n_rings = int(ring.max()) + 1
    print(f"points          {len(pts):,}")
    print(f"rings recovered {n_rings}   (expect 64 for HDL-64E)")

    # per-ring elevation, proves the rings are real and gives the beam table
    ring_elev = np.array([elev[ring == r].mean() if (ring == r).any() else np.nan
                          for r in range(n_rings)])
    finite = ring_elev[np.isfinite(ring_elev)]
    print(f"beam elevations {finite.max():+.2f} deg (top) .. {finite.min():+.2f} deg (bottom)")
    gaps = np.abs(np.diff(np.sort(finite)))
    print(f"beam spacing    median {np.median(gaps):.3f} deg  "
          f"min {gaps.min():.3f}  max {gaps.max():.3f}")

    # ---- pack ----------------------------------------------------------
    # xyz as int16 at 1 cm; intensity uint8; ring uint8
    xyz_q = np.clip(np.round(pts[:, :3] * 100), -32768, 32767).astype(np.int16)
    inten_q = np.clip(np.round(inten * 255), 0, 255).astype(np.uint8)
    ring_q = np.clip(ring, 0, 255).astype(np.uint8)

    blob = xyz_q.tobytes() + inten_q.tobytes() + ring_q.tobytes()
    b64 = base64.b64encode(blob).decode("ascii")
    (OUT / "payload.json.txt").write_text(b64)
    print(f"\npacked blob     {len(blob)/1e6:.2f} MB  -> base64 {len(b64)/1e6:.2f} MB")

    # ---- stats the page displays ---------------------------------------
    shells = []
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]:
        m = (rng >= lo) & (rng < hi)
        vol = (2 / 3) * np.pi * (hi**3 - lo**3)
        shells.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                       "vol": round(vol), "density": round(float(m.sum()) / vol, 4)})

    voxels = []
    for res in [0.05, 0.10, 0.20]:
        dense = int(160 / res) * int(160 / res) * int(10 / res)
        occ = len(np.unique((pts[:, :3] / res).astype(np.int32), axis=0))
        voxels.append({"res": res, "dense": dense, "occ": int(occ),
                       "pct": round(100 * occ / dense, 4)})

    meta = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "n": int(len(pts)),
        "nRings": n_rings,
        "sensorHeight": SENSOR_HEIGHT,
        "ringElev": [round(float(v), 3) if np.isfinite(v) else None for v in ring_elev],
        "stats": {
            "x": [float(x.min()), float(x.max())],
            "y": [float(y.min()), float(y.max())],
            "z": [float(z.min()), float(z.max())],
            "i": [float(inten.min()), float(inten.max())],
            "r": [float(rng.min()), float(rng.max())],
            "elev": [float(elev.min()), float(elev.max())],
        },
        "shells": shells,
        "voxels": voxels,
    }
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(f"meta.json       {len(json.dumps(meta))} bytes")

    # ---- camera image --------------------------------------------------
    try:
        from PIL import Image
        img_dir = path.parent.parent.parent / "image_02" / "data"
        img_path = sorted(img_dir.glob("*.png"))[FRAME]
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((1100, 1100))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72, optimize=True)
        jb64 = base64.b64encode(buf.getvalue()).decode("ascii")
        (OUT / "frame.jpg.txt").write_text(jb64)
        print(f"camera frame    {img_path.name}  {im.size}  -> base64 {len(jb64)/1e3:.0f} kB")
    except Exception as e:
        print(f"camera frame    skipped ({e})")


if __name__ == "__main__":
    main()
