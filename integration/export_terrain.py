"""Export the terrain layer, single-frame against accumulated, for the viewer.

Replays a sequence twice at each step:

    single       what one sweep alone can say about the ground right now
    accumulated  the world-anchored map, drift-corrected, seen from the same
                 pose at the same instant

Both are emitted in the CURRENT sensor frame so the viewer can put them side by
side without transforming anything, and both are scored by the identical
drivability code -- the only difference between the panels is how much evidence
went in. That is the whole claim, so nothing else may differ.

Cells go out as int16 grid indices plus a class byte and a quantised height.
The map is sparse and mostly unchanged between frames, but it is re-emitted per
frame rather than diffed: a diff would have to be replayed in order, and the
viewer lets you scrub.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

import grid25 as g
import accumulate as A
import terrain_cells as tc

RES = g.res0 * (1 << tc.DRIVE_LEVEL)      # drivability lattice, metres
QZ = 100.0                                 # int16 units per metre of height
NAMES = ["drivable", "marginal", "non-drivable", "unknown"]
BANDS = ((0, 10), (10, 20), (20, 30), (30, 50))


def ground_labels(pts, remove_ground):
    """Geometric ground segmentation -- the deployed path, no labels needed."""
    isg, _, _ = remove_ground(pts.astype(np.float32))
    return np.where(isg, g.road, g.other)


def local(cells, T_w_velo, use_world):
    """Cell centres in the current sensor frame, plus class and height.

    The accumulated map is world-anchored, so its cells must be brought back
    into the sensor frame to be drawn beside the single-frame one. A single
    frame quantised in sensor coordinates is already there.
    """
    cx = (cells["ix"] + 0.5) * RES
    cy = (cells["iy"] + 0.5) * RES
    sxy = (T_w_velo[0, 3], T_w_velo[1, 3]) if use_world else (0.0, 0.0)
    _, cls, h = tc.cell_drivability(cells, sxy, res=RES)

    if use_world:
        inv = np.linalg.inv(T_w_velo)
        hz = np.nan_to_num(h, nan=0.0)
        p = A.apply(inv, np.stack([cx, cy, hz], 1))
        cx, cy, h = p[:, 0], p[:, 1], p[:, 2]
    else:
        h = np.nan_to_num(h, nan=0.0)
    return cx, cy, cls, h


def pack(cx, cy, cls, h, rmax):
    keep = (np.hypot(cx, cy) <= rmax) & (cls != tc.UNKNOWN)
    ix = np.round(cx[keep] / RES).astype(np.int16)
    iy = np.round(cy[keep] / RES).astype(np.int16)
    hz = np.clip(np.round(h[keep] * QZ), -32768, 32767).astype(np.int16)
    return ix, iy, cls[keep].astype(np.uint8), hz


def summarise(cx, cy, cls, rmax, ahead=None):
    """Class mix by range band.

    `ahead` splits the map at the vehicle's own axis. This matters and must not
    be averaged away: the trust gate means only returns from inside 20 m are
    ever ingested, so the accumulated map extends BEHIND the vehicle much
    further than in front of it. Its gain is on ground already driven past,
    remembered densely -- not on ground not yet reached. Reporting one blended
    number would read as extra forward reach, which the method does not have.
    """
    r = np.hypot(cx, cy)
    sel = np.ones(len(r), bool) if ahead is None else (
        (cx > 0) if ahead else (cx <= 0))
    out = []
    for lo, hi in BANDS:
        m = sel & (r >= lo) & (r < hi)
        n = int(m.sum())
        out.append({"band": f"{lo}-{hi}", "n": n,
                    "f": [round(float((cls[m] == k).sum()) / n, 4)
                          if n else 0.0 for k in range(4)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="directory of NN_FFFFFF.bin sweeps")
    ap.add_argument("--seq", default="00")
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--pnd", type=Path, required=True,
                    help="pointnet-det/src, for the ground segmenter")
    ap.add_argument("--max-frame", type=int, default=40)
    ap.add_argument("--rmax", type=float, default=50.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.pnd))
    from pnd.ground import remove_ground

    T = A.load_poses(a.poses, a.calib)
    files = sorted(a.cache.glob(f"{a.seq}_*.bin"))
    seq = [(int(f.stem.split("_")[1]), f) for f in files]
    seq = [(i, f) for i, f in seq if i <= a.max_frame]
    if not seq:
        raise SystemExit("no frames")
    print(f"{len(seq)} frames: {[i for i, _ in seq]}")

    wm = A.WorldMap()
    frames = []
    bs, ba = bytearray(), bytearray()

    for i, f in seq:
        pts = np.fromfile(f, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        lab = ground_labels(pts, remove_ground)

        one = tc.coarsen(g.quantise(pts[:, 0], pts[:, 1], pts[:, 2], lab))
        cx1, cy1, c1, h1 = local(one, T[i], use_world=False)

        wm.ingest(pts, lab, T[i])
        acc = tc.coarsen(wm.c)
        cx2, cy2, c2, h2 = local(acc, T[i], use_world=True)

        ix1, iy1, k1, z1 = pack(cx1, cy1, c1, h1, a.rmax)
        ix2, iy2, k2, z2 = pack(cx2, cy2, c2, h2, a.rmax)
        # grouped by field, never interleaved: an odd cell count would leave
        # the next frame's Int16Array on an odd byte offset, which throws
        bs += ix1.tobytes() + iy1.tobytes() + z1.tobytes() + k1.tobytes()
        ba += ix2.tobytes() + iy2.tobytes() + z2.tobytes() + k2.tobytes()

        frames.append({
            "i": i,
            "ns": int(len(ix1)), "na": int(len(ix2)),
            "dz": round(float(wm.dz) * 100, 2),
            "pose": [round(float(v), 3) for v in T[i][:3, 3]],
            "ss": summarise(cx1, cy1, c1, a.rmax, ahead=True),
            "sa": summarise(cx2, cy2, c2, a.rmax, ahead=True),
            "ssb": summarise(cx1, cy1, c1, a.rmax, ahead=False),
            "sab": summarise(cx2, cy2, c2, a.rmax, ahead=False),
            "xr": [round(float(cx2.min()), 1), round(float(cx2.max()), 1)],
        })
        print(f"  frame {i:>4}: single {len(ix1):>6,} cells   "
              f"accum {len(ix2):>6,} cells   dz {wm.dz*100:+6.2f} cm")

    payload = {
        "res": RES, "qz": QZ, "rmax": a.rmax,
        "classes": NAMES,
        "bands": [f"{lo}-{hi}" for lo, hi in BANDS],
        "params": {"rough": tc.MAX_ROUGH_M, "step": tc.MAX_STEP_M,
                   "slope": tc.MAX_SLOPE_DEG, "trust": wm.trust,
                   "level": tc.DRIVE_LEVEL, "fine": g.res0},
        "frames": frames,
        "single": base64.b64encode(bytes(bs)).decode("ascii"),
        "accum": base64.b64encode(bytes(ba)).decode("ascii"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload))
    print(f"\nwrote {a.out}  {a.out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
