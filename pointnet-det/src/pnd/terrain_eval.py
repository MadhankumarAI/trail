"""
Measure the geometric drivability estimate against KITTI ROAD ground truth.

    python -m pnd.terrain_eval --max-frames 100

terrain.py was previously only eyeballed -- a bird's-eye plot showed a road-shaped
corridor and that was taken as working. A corridor-shaped blob is not a number,
so this scores it.

HOW THE LABELS ARE OBTAINED
---------------------------
KITTI ROAD annotates the road in *image* space, not on the point cloud. Each
gt_image_2 PNG encodes:

    [255, 0, 255]   valid and road
    [255,   0, 0]   valid, not road
    [  0,   0, 0]   not annotated

So every LiDAR point is projected into the image with Tr_velo_to_cam / R0_rect /
P2 and takes the label of the pixel it lands on. Points that fall outside the
image, behind the camera, or on an unannotated pixel are excluded -- absence of
a label is not evidence of "not road".

WHAT IS BEING COMPARED
----------------------
Ground truth is a *semantic* label: is this the road surface. The prediction is
a *geometric* one: is this patch flat, smooth and continuous enough to drive on.
They are not the same question and the gap is the interesting part -- a flat
driveway apron or a smooth pedestrian plaza is geometrically drivable and is not
road, and will show up here as a false positive. PS 26053 asks to "distinguish
between drivable surfaces and non-drivable terrain", which is the geometric
question, so those are not necessarily errors. They are reported rather than
hidden.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .ground import remove_ground
from .kitti import Calib
from .terrain import DRIVABLE, MARGINAL, NON_DRIVABLE, UNKNOWN, analyse

ROOT = Path(__file__).resolve().parents[2]
ROAD = ROOT / "data" / "kitti_road"


def frame_ids(split: str = "training"):
    velo = ROAD / split / "velodyne"
    return sorted(p.stem for p in velo.glob("*.bin"))


def load_gt(fid: str, split: str = "training"):
    """Road mask and valid mask for one frame, as image-space boolean arrays."""
    from PIL import Image
    cat, num = fid.rsplit("_", 1)
    p = ROAD / "data_road" / split / "gt_image_2" / f"{cat}_road_{num}.png"
    if not p.exists():
        return None, None
    im = np.array(Image.open(p))
    return im[:, :, 2] > 0, im[:, :, 0] > 0        # road, valid


def score_frame(fid: str, cfg_thresh: dict, split: str = "training",
                grid: dict | None = None):
    """Confusion counts for one frame, plus a range breakdown."""
    velo = ROAD / split / "velodyne" / f"{fid}.bin"
    calib_p = ROAD / "data_road" / split / "calib" / f"{fid}.txt"
    if not velo.exists() or not calib_p.exists():
        return None

    road, valid = load_gt(fid, split)
    if road is None:
        return None
    H, W = road.shape

    pts = np.fromfile(velo, dtype=np.float32).reshape(-1, 4)
    calib = Calib.from_file(calib_p)

    is_ground, agl, stats = remove_ground(pts[:, :3], **(grid or {}))
    res = analyse(pts[:, :3], is_ground, stats, **cfg_thresh)
    pred = res["point_cls"]

    # project every point into the image and take the pixel's label
    cam = calib.velo_to_cam(pts[:, :3])
    uv = calib.project_to_image(pts[:, :3])
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inside = (cam[:, 2] > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # score only ground points that carry a label: a non-ground point is not a
    # drivability question, and an unannotated pixel is not a negative
    m = inside & is_ground
    if m.sum() < 50:
        return None
    uu, vv = u[m], v[m]
    m_valid = valid[vv, uu]
    if m_valid.sum() < 50:
        return None

    gt_road = road[vv, uu][m_valid]
    p_drive = (pred[m][m_valid] == DRIVABLE)
    rng = np.linalg.norm(pts[m][m_valid][:, :2], axis=1)

    def counts(sel):
        g, p = gt_road[sel], p_drive[sel]
        return np.array([(g & p).sum(), (~g & p).sum(),
                         (g & ~p).sum(), (~g & ~p).sum()], np.int64)

    out = {"all": counts(np.ones(len(gt_road), bool))}
    for lo, hi in [(0, 10), (10, 20), (20, 40)]:
        s = (rng >= lo) & (rng < hi)
        if s.sum() > 20:
            out[f"{lo}-{hi}"] = counts(s)
    return out


def prf(c):
    tp, fp, fn, tn = c
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    acc = (tp + tn) / max(c.sum(), 1)
    return prec, rec, f1, iou, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--max-step", type=float, default=0.10)
    ap.add_argument("--max-rough", type=float, default=0.020)
    ap.add_argument("--sweep", action="store_true",
                    help="try a grid of thresholds instead of one setting")
    a = ap.parse_args()

    ids = frame_ids()
    if a.max_frames:
        ids = ids[:a.max_frames]
    if not ids:
        raise SystemExit(f"no scans under {ROAD}")

    settings = [(a.max_slope, a.max_step, a.max_rough)]
    if a.sweep:
        settings = [(s, st, r)
                    for s in (10.0, 15.0, 20.0)
                    for st in (0.08, 0.12, 0.20)
                    for r in (0.04, 0.06, 0.10)]

    print(f"KITTI ROAD, {len(ids)} frames, "
          f"{len(settings)} threshold setting(s)\n")

    best = None
    for (slope, step, rough) in settings:
        thr = {"max_slope_deg": slope, "max_step": step, "max_rough": rough}
        tot = np.zeros(4, np.int64)
        by_r = {}
        bar = tqdm(ids, ncols=78, leave=False) if not a.sweep else ids
        for fid in bar:
            r = score_frame(fid, thr)
            if r is None:
                continue
            tot += r["all"]
            for k, v in r.items():
                if k != "all":
                    by_r[k] = by_r.get(k, np.zeros(4, np.int64)) + v
        p, rc, f1, iou, acc = prf(tot)
        tag = f"slope<{slope:>4.0f}  step<{step:.2f}  rough<{rough:.2f}"
        print(f"  {tag}   P {p:.3f}  R {rc:.3f}  F1 {f1:.3f}  "
              f"IoU {iou:.3f}  acc {acc:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, tag, tot, by_r)

    if a.sweep:
        print(f"\nbest: {best[1]}   F1 {best[0]:.3f}")
    tot, by_r = best[2], best[3]

    print("\nconfusion (ground points with a road label)")
    tp, fp, fn, tn = tot
    print(f"  road & predicted drivable      {tp:>10,}")
    print(f"  not road & predicted drivable  {fp:>10,}   <- flat non-road")
    print(f"  road & predicted not drivable  {fn:>10,}")
    print(f"  not road & not drivable        {tn:>10,}")

    if by_r:
        print("\nby range")
        print(f"  {'band':<10}{'P':>8}{'R':>8}{'F1':>8}{'IoU':>8}{'points':>12}")
        for k in sorted(by_r, key=lambda s: int(s.split('-')[0])):
            p, rc, f1, iou, _ = prf(by_r[k])
            print(f"  {k+' m':<10}{p:>8.3f}{rc:>8.3f}{f1:>8.3f}{iou:>8.3f}"
                  f"{by_r[k].sum():>12,}")


if __name__ == "__main__":
    main()
