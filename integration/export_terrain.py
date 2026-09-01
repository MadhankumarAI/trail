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


def obstacle_cells(cells, T_w_velo, rmax):
    """Static-obstacle cells of the 2.5D map, as (ix, iy, height-above-ground).

    An obstacle in a height field is not a separate object list: it is a column
    whose top stands above the local ground. zmax gives that top, and the cell
    already knows its own ground height, so the obstacle layer costs one
    subtraction and no new structure.
    """
    n = cells["hist"][:, g.bldg]
    sel = n >= 3
    if not sel.any():
        e16 = np.zeros(0, np.int16)
        return e16, e16, e16

    c = {k: v[sel] for k, v in cells.items()}
    cx = (c["ix"] + 0.5) * RES
    cy = (c["iy"] + 0.5) * RES
    top = c["zmax"]
    base = np.where(c["ng"] > 0, c["gsum"] / np.maximum(c["ng"], 1), np.nan)

    inv = np.linalg.inv(T_w_velo)
    p = A.apply(inv, np.stack([cx, cy, top], 1))
    q = A.apply(inv, np.stack([cx, cy, np.nan_to_num(base, nan=0.0)], 1))
    # height above the ground under it where that is known, else above the
    # sensor's own road plane -- never a raw z, which is measured from the laser
    hgt = np.where(np.isfinite(base), p[:, 2] - q[:, 2], p[:, 2] + 1.73)

    keep = (np.hypot(p[:, 0], p[:, 1]) <= rmax) & (hgt > 0.30) & (hgt < 8.0)
    return (np.round(p[keep, 0] / RES).astype(np.int16),
            np.round(p[keep, 1] / RES).astype(np.int16),
            np.clip(np.round(hgt[keep] * QZ), 0, 32767).astype(np.int16))


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
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="detector checkpoint; adds the object layer")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.pnd))
    from pnd.ground import remove_ground

    # ---- optional object layer -------------------------------------- #
    model = cfg = process = None
    if a.ckpt:
        import torch
        from pnd.config import Config
        from pnd.model import build
        from pnd.simulate import process, CLS_STATIC
        cfg = Config.load()
        ck = torch.load(a.ckpt, map_location=cfg.device, weights_only=False)
        for k in ("canon", "in_ch", "width", "num_classes", "dropout",
                  "n_points", "cluster_voxel", "min_cluster_pts",
                  "max_cluster_pts", "ground_thresh", "max_range"):
            if k in ck["cfg"]:
                setattr(cfg, k, ck["cfg"][k])
        model = build(cfg).to(cfg.device)
        model.load_state_dict(ck["model"])
        model.eval()
        np.random.seed(0)
        print(f"detector {a.ckpt.name}  canon={cfg.canon}  device={cfg.device}  "
              f"F1={ck.get('metrics', {}).get('f1_fg', 0):.4f}")

    T = A.load_poses(a.poses, a.calib)
    files = sorted(a.cache.glob(f"{a.seq}_*.bin"))
    seq = [(int(f.stem.split("_")[1]), f) for f in files]
    seq = [(i, f) for i, f in seq if i <= a.max_frame]
    if not seq:
        raise SystemExit("no frames")
    print(f"{len(seq)} frames: {[i for i, _ in seq]}")

    wm = A.WorldMap()
    frames = []
    bs, ba, bo = bytearray(), bytearray(), bytearray()

    for i, f in seq:
        raw = np.fromfile(f, np.float32).reshape(-1, 4).astype(np.float64)
        pts = raw[:, :3]
        lab = ground_labels(pts, remove_ground)

        # ---- objects ------------------------------------------------- #
        # Dynamic detections are kept OUT of the persistent map. A car driving
        # past would otherwise lay down a solid wall of evidence along its whole
        # path, and that wall would be scored as terrain. Static obstacles do
        # accumulate: they are static, so more looks at them is simply better.
        boxes, moving = [], None
        if model is not None:
            dcls, boxes, _t, _n, _sc, _sh = process(raw, cfg, model,
                                                    cfg.device, None)
            moving = dcls >= CLS_STATIC + 1          # Car / Pedestrian / Cyclist
            static = dcls == CLS_STATIC
            lab = np.where(static, g.bldg, lab)

        one = tc.coarsen(g.quantise(pts[:, 0], pts[:, 1], pts[:, 2], lab))
        cx1, cy1, c1, h1 = local(one, T[i], use_world=False)

        wm.ingest(pts, lab, T[i], moving=moving)
        acc = tc.coarsen(wm.c)
        cx2, cy2, c2, h2 = local(acc, T[i], use_world=True)

        ix1, iy1, k1, z1 = pack(cx1, cy1, c1, h1, a.rmax)
        ix2, iy2, k2, z2 = pack(cx2, cy2, c2, h2, a.rmax)
        ox, oy, oz = obstacle_cells(acc, T[i], a.rmax)
        # grouped by field, never interleaved: an odd cell count would leave
        # the next frame's Int16Array on an odd byte offset, which throws
        bs += ix1.tobytes() + iy1.tobytes() + z1.tobytes() + k1.tobytes()
        ba += ix2.tobytes() + iy2.tobytes() + z2.tobytes() + k2.tobytes()
        bo += ox.tobytes() + oy.tobytes() + oz.tobytes()

        frames.append({
            "i": i,
            "ns": int(len(ix1)), "na": int(len(ix2)), "no": int(len(ox)),
            "box": boxes,
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
        "obst": base64.b64encode(bytes(bo)).decode("ascii"),
        "objcls": ["", "Car", "Pedestrian", "Cyclist"],
        "hasobj": model is not None,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload))
    print(f"\nwrote {a.out}  {a.out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
