"""
Turn raw scans into labelled cluster proposals, once, and cache them.

Pipeline per frame:
    load scan -> keep camera FOV -> remove ground -> cluster -> label each
    cluster against the GT boxes -> sample a fixed number of points -> store

THE TWO LABELLING TRAPS
-----------------------
Both of these silently poison training and neither raises an error.

1. KITTI only annotates the front camera frustum. A cluster behind the vehicle
   is *unlabelled*, not background. Calling it background teaches the network
   that cars are background. Everything outside `calib.fov_mask` is dropped.

2. `DontCare` marks regions the annotators refused to label, and Van / Truck /
   Person_sitting / Tram / Misc are real objects that are not in our class set.
   A cluster overlapping any of them is dropped, not labelled background.

Proposals that overlap a target box only partially (between `bg_max_pts` and
`fg_iou_pts`) are also dropped: they are neither clean foreground nor honest
background, and including them mostly adds label noise.

    python -m pnd.proposals --max-frames 1500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .cluster import cluster_points
from .config import Config
from .ground import remove_ground
from .kitti import (CLASSES, Calib, IGNORE_TYPES, points_in_box, read_labels,
                    read_velodyne)


def frame_proposals(frame: str, cfg: Config):
    """All proposals for one frame. Returns dict of arrays, or None if unusable."""
    root = cfg.data_root / "training"
    velo_p = root / "velodyne" / f"{frame}.bin"
    if not velo_p.exists():
        return None

    pts = read_velodyne(velo_p)
    calib = Calib.from_file(root / "calib" / f"{frame}.txt")
    objs = read_labels(root / "label_2" / f"{frame}.txt")

    # trap 1: only the annotated frustum
    r = np.linalg.norm(pts[:, :2], axis=1)
    keep = calib.fov_mask(pts[:, :3]) & (r < cfg.max_range) & (r > 1.0)
    pts = pts[keep]
    if len(pts) < 500:
        return None

    is_ground, agl, _ = remove_ground(pts[:, :3], thresh=cfg.ground_thresh)
    fg = ~is_ground
    if fg.sum() < 50:
        return None

    obj_pts = pts[fg]
    obj_agl = agl[fg]
    lab = cluster_points(obj_pts[:, :3], voxel=cfg.cluster_voxel,
                         min_points=cfg.min_cluster_pts,
                         max_points=cfg.max_cluster_pts)
    n_cl = int(lab.max()) + 1
    if n_cl <= 0:
        return None

    # precompute box masks over the non-ground points
    targets, ignores = [], []
    for o in objs:
        if o.type in IGNORE_TYPES:
            if o.type == "DontCare":
                continue                      # 2D only; handled via bbox below
            ignores.append(o)
        elif o.is_target:
            targets.append(o)

    tgt_masks = [points_in_box(obj_pts[:, :3], o.center_velo(calib),
                               o.dims_velo(), o.yaw_velo(), margin=0.1)
                 for o in targets]
    ign_masks = [points_in_box(obj_pts[:, :3], o.center_velo(calib),
                               o.dims_velo(), o.yaw_velo(), margin=0.2)
                 for o in ignores]

    P, C = [], []
    for k in range(n_cl):
        m = lab == k
        n = int(m.sum())
        if n < cfg.min_cluster_pts:
            continue

        # trap 2: overlapping an ignored class -> drop entirely
        if any((im & m).sum() / n > cfg.bg_max_pts for im in ign_masks):
            continue

        best_j, best_frac = -1, 0.0
        for j, tm in enumerate(tgt_masks):
            f = (tm & m).sum() / n
            if f > best_frac:
                best_frac, best_j = f, j

        if best_frac >= cfg.fg_iou_pts:
            o = targets[best_j]
            cls = o.class_id
            ctr = o.center_velo(calib)
            dims = o.dims_velo()
            yaw = o.yaw_velo()
        elif best_frac <= cfg.bg_max_pts:
            cls = 0
            ctr = obj_pts[m, :3].mean(0)
            ext = obj_pts[m, :3].max(0) - obj_pts[m, :3].min(0)
            dims = np.maximum(ext, 0.2)
            yaw = 0.0
        else:
            continue                         # ambiguous partial overlap

        idx = np.flatnonzero(m)
        sel = (np.random.choice(idx, cfg.n_points, replace=False)
               if len(idx) >= cfg.n_points
               else np.random.choice(idx, cfg.n_points, replace=True))

        q = obj_pts[sel]
        P.append(np.column_stack([q[:, 0], q[:, 1], q[:, 2],
                                  q[:, 3], obj_agl[sel]]).astype(np.float32))
        C.append(np.concatenate([[cls], ctr, dims, [yaw], [n]]).astype(np.float32))

    if not P:
        return None
    return {"points": np.stack(P), "meta": np.stack(C)}


def build_cache(cfg: Config, shard_size: int = 500) -> None:
    root = cfg.data_root / "training" / "velodyne"
    frames = sorted(p.stem for p in root.glob("*.bin"))
    if cfg.max_frames:
        frames = frames[:cfg.max_frames]

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"building proposals from {len(frames):,} frames")
    print(f"  -> {cfg.cache_dir}")

    buf_p, buf_m, buf_f = [], [], []
    shard, n_total = 0, 0
    t0 = time.time()
    counts = np.zeros(len(CLASSES), np.int64)
    n_empty = 0

    def flush():
        nonlocal shard, buf_p, buf_m, buf_f
        if not buf_p:
            return
        np.savez_compressed(
            cfg.cache_dir / f"shard_{shard:04d}.npz",
            points=np.concatenate(buf_p),
            meta=np.concatenate(buf_m),
            frame=np.concatenate(buf_f),
        )
        shard += 1
        buf_p, buf_m, buf_f = [], [], []

    for i, f in enumerate(tqdm(frames, ncols=78, desc="frames")):
        out = frame_proposals(f, cfg)
        if out is None:
            n_empty += 1
            continue
        buf_p.append(out["points"])
        buf_m.append(out["meta"])
        buf_f.append(np.full(len(out["meta"]), int(f), np.int32))
        n_total += len(out["meta"])
        for c in out["meta"][:, 0].astype(int):
            counts[c] += 1
        if (i + 1) % shard_size == 0:
            flush()
    flush()

    el = time.time() - t0
    print(f"\n{n_total:,} proposals in {shard} shards  ({el:.0f}s, "
          f"{1000*el/max(len(frames),1):.0f} ms/frame)")
    print(f"frames yielding nothing: {n_empty}")
    print("\nclass balance:")
    for i, nm in enumerate(CLASSES):
        print(f"  {nm:<12}{counts[i]:>9,}  {100*counts[i]/max(n_total,1):>6.2f}%")
    if counts[1:].sum():
        print(f"\nbackground : foreground = "
              f"{counts[0]/max(counts[1:].sum(),1):.1f} : 1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=500)
    a = ap.parse_args()
    cfg = Config.load(a.config, max_frames=a.max_frames)
    np.random.seed(cfg.seed)
    build_cache(cfg, a.shard_size)


if __name__ == "__main__":
    main()
