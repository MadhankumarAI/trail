"""
End-to-end detection evaluation.

    python -m pnd.evaluate --canon pca2_yaw

Reports three things, deliberately kept separate because they fail for
different reasons and conflating them hides where the system is actually weak.

1. CLUSTER RECALL -- the ceiling.
   For every ground-truth object, did the ground-removal + clustering stage
   produce any cluster containing at least `fg_iou_pts` of its points? An object
   that fails here is invisible to the network: no classifier, however good, can
   recover it. This bounds everything below.

   The training F1 numbers are conditional on this stage succeeding, because
   proposals only entered the cache if they matched a box. Quoting F1 as
   "detection accuracy" without this number overstates the system.

2. CLASSIFIER PERFORMANCE on the proposals that survived clustering.

3. 3D AP -- the two multiplied together, in the form the KITTI benchmark uses,
   split by its easy / moderate / hard difficulty rules.

Also breaks results down by range, since the whole premise of the project is
that behaviour changes with distance.

IoU note: this uses BEV rotated-rectangle IoU multiplied by the height overlap
ratio. That is the standard approximation and it matches the official metric
closely for upright road objects, which never roll or pitch appreciably.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .cluster import cluster_points
from .boxes import decode_heading
from .config import Config
from .dataset import ANCHORS
from .ground import remove_ground
from .kitti import (CLASSES, Calib, IGNORE_TYPES, points_in_box, read_labels,
                    read_velodyne)
from .model import build

# KITTI difficulty, from the 2D box height in pixels plus occlusion/truncation
DIFFICULTY = [
    ("easy",     40, 0, 0.15),
    ("moderate", 25, 1, 0.30),
    ("hard",     25, 2, 0.50),
]
# KITTI's official thresholds: 0.7 for the car class, 0.5 for the two small
# ones. Van and Truck are not in the official benchmark, so the choice is ours
# -- 0.7, the same as Car, because they are vehicles of comparable size and
# scoring them at 0.5 would flatter them against Car in the same table.
IOU_THRESH = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5,
              "Van": 0.7, "Truck": 0.7}


# --------------------------------------------------------------------------- #
# rotated box IoU
# --------------------------------------------------------------------------- #
def _corners_2d(cx, cy, l, w, yaw):
    """Four corners, counter-clockwise.

    The winding matters: _clip's inside-test assumes counter-clockwise, and a
    clockwise polygon makes every vertex test as outside, so the intersection
    comes back empty and every IoU is 0.0 -- including a box against itself.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([l, l, -l, -l]) / 2.0
    dy = np.array([-w, w, w, -w]) / 2.0
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=1)


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(poly):
    return poly if _signed_area(poly) >= 0 else poly[::-1]


def _clip(subject, clipper):
    """Sutherland-Hodgman: clip a convex polygon by another convex polygon."""
    subject, clipper = _ccw(subject), _ccw(clipper)
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a
        if not out.size:
            return out
        inp, res = out, []
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            sq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            if sp >= 0:
                res.append(p)
            if (sp >= 0) != (sq >= 0):
                d = sp - sq
                if abs(d) > 1e-12:
                    res.append(p + (q - p) * (sp / d))
        out = np.array(res) if res else np.empty((0, 2))
    return out


def _area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box_iou_3d(a, b) -> float:
    """a, b = (cx, cy, cz, l, w, h, yaw) in velodyne coordinates."""
    pa = _corners_2d(a[0], a[1], a[3], a[4], a[6])
    pb = _corners_2d(b[0], b[1], b[3], b[4], b[6])
    # counter-clockwise for the clipper
    if _area(pa) == 0 or _area(pb) == 0:
        return 0.0
    inter2d = _area(_clip(pa, pb))
    if inter2d <= 0:
        return 0.0
    za0, za1 = a[2] - a[5] / 2, a[2] + a[5] / 2
    zb0, zb1 = b[2] - b[5] / 2, b[2] + b[5] / 2
    oz = max(0.0, min(za1, zb1) - max(za0, zb0))
    inter = inter2d * oz
    union = a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- #
def val_frame_ids(cfg: Config):
    """Reproduce dataset.py's frame-level split exactly."""
    frames = []
    for s in sorted(cfg.cache_dir.glob("shard_*.npz")):
        frames.append(np.load(s)["frame"])
    uf = np.unique(np.concatenate(frames))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(uf)
    n_val = max(int(len(uf) * cfg.val_frac), 1)
    return sorted(int(f) for f in uf[:n_val])


def difficulties_of(o) -> set:
    """Every KITTI difficulty bucket this object belongs to.

    The buckets are NESTED, not exclusive: an object meeting the easy criteria
    also counts towards moderate and hard. Assigning each object to only the
    easiest bucket it qualifies for leaves 'moderate' holding just the objects
    that FAIL easy, which is a small and unnaturally hard subset -- it produced
    Cyclist easy 73.12 / moderate 0.00 / hard 2.63, and makes the numbers
    incomparable to any published result.
    """
    h = o.bbox2d[3] - o.bbox2d[1]
    out = set()
    for name, min_h, max_occ, max_tr in DIFFICULTY:
        if h >= min_h and o.occlusion <= max_occ and o.truncation <= max_tr:
            out.add(name)
    return out


@torch.no_grad()
def run_frame(frame: str, cfg: Config, model, device, opt=None):
    """Full pipeline on one frame.

    `opt` carries the false-positive controls so each can be measured alone:
        reject_bg   drop clusters whose argmax over ALL classes is background
        score_mode  "class" = p(class); "fg" = 1 - p(background)
        nms         per-class 3D IoU suppression threshold, 0 disables
        min_score   confidence floor
    """
    opt = opt or {}
    from .bench_canon import pca2_batch, pca3_batch
    from .dataset import _rot_z

    root = cfg.data_root / "training"
    pts = read_velodyne(root / "velodyne" / f"{frame}.bin")
    calib = Calib.from_file(root / "calib" / f"{frame}.txt")
    objs = [o for o in read_labels(root / "label_2" / f"{frame}.txt")
            if o.is_target]

    r = np.linalg.norm(pts[:, :2], axis=1)
    keep = calib.fov_mask(pts[:, :3]) & (r < cfg.max_range) & (r > 1.0)
    pts = pts[keep]
    if len(pts) < 500:
        return [], objs, {}, calib

    is_ground, agl, _ = remove_ground(pts[:, :3], thresh=cfg.ground_thresh)
    op, oa = pts[~is_ground], agl[~is_ground]
    if len(op) < 50:
        return [], objs, {}, calib

    lab = cluster_points(op[:, :3], voxel=cfg.cluster_voxel,
                         min_points=cfg.min_cluster_pts,
                         max_points=cfg.max_cluster_pts)
    n_cl = int(lab.max()) + 1
    if n_cl <= 0:
        return [], objs, {}, calib

    # ---- cluster recall: did any cluster capture this object? ---------- #
    hits = {}
    for oi, o in enumerate(objs):
        m = points_in_box(op[:, :3], o.center_velo(calib), o.dims_velo(),
                          o.yaw_velo(), margin=0.1)
        best = 0.0
        for k in range(n_cl):
            c = lab == k
            n = int(c.sum())
            if n:
                best = max(best, (m & c).sum() / n)
        hits[oi] = best >= cfg.fg_iou_pts

    # ---- build proposals ------------------------------------------------ #
    P, ctrs = [], []
    for k in range(n_cl):
        m = lab == k
        idx = np.flatnonzero(m)
        if len(idx) < cfg.min_cluster_pts:
            continue
        sel = (np.random.choice(idx, cfg.n_points, replace=False)
               if len(idx) >= cfg.n_points
               else np.random.choice(idx, cfg.n_points, replace=True))
        q = op[sel]
        P.append(np.column_stack([q[:, 0], q[:, 1], q[:, 2], q[:, 3], oa[sel]]))
        ctrs.append(op[m, :3].mean(0))
    if not P:
        return [], objs, hits, calib

    arr = np.stack(P).astype(np.float64)
    B, PN = arr.shape[0], arr.shape[1]
    xyz = arr[:, :, :3]
    inten, aglv = arr[:, :, 3], arr[:, :, 4]
    rng_raw = np.linalg.norm(xyz, axis=2)

    flat = np.ascontiguousarray(xyz.reshape(-1, 3))
    offs = (np.arange(B + 1) * PN).astype(np.int64)
    Rc = np.zeros((B, 3, 3)); tc = np.zeros((B, 3)); lam = np.zeros((B, 3))
    if cfg.canon in ("none", "tnet3"):
        tc = xyz.mean(1); Rc = np.repeat(np.eye(3)[None], B, 0)
    elif cfg.canon == "pca3_skew":
        pca3_batch(flat, offs, Rc, tc, lam)
    else:
        yc = np.zeros(B); pca2_batch(flat, offs, yc)
        tc = xyz.mean(1); Rc = _rot_z(-yc)

    cen = xyz - tc[:, None, :]
    xc = np.einsum("bij,bpj->bpi", Rc, cen)
    scale = np.maximum(np.linalg.norm(xc, axis=2).max(1), 1e-6)
    xc /= scale[:, None, None]

    feats = np.concatenate([xc, (aglv / 3.0)[:, :, None],
                            (rng_raw / cfg.max_range)[:, :, None],
                            inten[:, :, None]], axis=2).transpose(0, 2, 1)
    x = torch.from_numpy(feats).float().to(device)
    if cfg.canon == "pca4_ensemble":
        cps = []
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            f = x.clone(); f[:, 0] *= sx; f[:, 1] *= sy; cps.append(f)
        x = torch.stack(cps, 1)

    with torch.autocast(device, dtype=getattr(torch, cfg.amp_dtype),
                        enabled=cfg.amp_dtype is not None):
        out = model(x)

    prob = out["logits"].float().softmax(1).cpu().numpy()
    dc = out["center"].float().cpu().numpy()
    sl = out["size_log"].float().cpu().numpy()
    hb = out["head_bin"].float().argmax(1)
    hr = out["head_res"].float().gather(1, hb.unsqueeze(1)).squeeze(1)
    yaw_pred = decode_heading(hb.float(), hr).cpu().numpy()

    dets = []
    yaw_off = np.arctan2(Rc[:, 1, 0], Rc[:, 0, 0])
    reject_bg = opt.get("reject_bg", False)
    score_mode = opt.get("score_mode", "fg")
    min_score = opt.get("min_score", 0.05)

    for i in range(B):
        # Argmax over ALL classes, background included. The original took
        # argmax over prob[1:] only, so every cluster emitted a detection even
        # when the classifier confidently said background -- roughly 23 false
        # positives per frame against ~3 real objects. (The old `if c == 0`
        # guard was dead code: an argmax over prob[1:] can never return 0.)
        top = int(prob[i].argmax())
        if reject_bg and top == 0:
            continue
        c = top if top > 0 else int(prob[i, 1:].argmax()) + 1

        # "class" ranks by p(class); "fg" by total foreground mass, which ranks
        # better when the model is split between two foreground classes.
        score = float(1.0 - prob[i, 0]) if score_mode == "fg" else float(prob[i, c])
        if score < min_score:
            continue

        ctr = Rc[i].T @ (dc[i] * scale[i]) + tc[i]
        dims = np.exp(sl[i]) * ANCHORS[c]
        yaw = float(yaw_pred[i] - yaw_off[i])
        dets.append({"cls": c, "score": score,
                     "box": np.array([ctr[0], ctr[1], ctr[2],
                                      dims[0], dims[1], dims[2], yaw])})

    # Per-class 3D NMS. Nothing otherwise stops two adjacent clusters both
    # claiming the same car.
    thr = opt.get("nms", 0.0)
    if thr > 0 and dets:
        kept = []
        for c in set(d["cls"] for d in dets):
            cand = sorted([d for d in dets if d["cls"] == c],
                          key=lambda d: -d["score"])
            while cand:
                best = cand.pop(0)
                kept.append(best)
                cand = [d for d in cand
                        if box_iou_3d(best["box"], d["box"]) < thr]
        dets = kept
    return dets, objs, hits, calib


def gt_boxes_velo(objs, calib):
    """(N, 7) ground-truth boxes in velodyne coords: cx cy cz l w h yaw."""
    if not objs:
        return np.zeros((0, 7))
    return np.stack([np.concatenate([o.center_velo(calib), o.dims_velo(),
                                     [o.yaw_velo()]]) for o in objs])


def average_precision(recs, precs) -> float:
    """VOC-style AP with a monotonic precision envelope."""
    mrec = np.concatenate([[0.0], recs, [1.0]])
    mpre = np.concatenate([[0.0], precs, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--canon", default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="evaluate on a subset, for a quick check")
    ap.add_argument("--reject-bg", action="store_true",
                    help="drop clusters classified as background. MEASURED "
                         "HARMFUL - leave off. In AP a rejected cluster can "
                         "never match a ground-truth box, so every rejection "
                         "is a guaranteed miss when wrong, while a kept "
                         "low-scored cluster sorts below the true positives "
                         "and costs almost nothing.")
    ap.add_argument("--score-mode", choices=["class", "fg"], default="fg",
                    help="fg = 1 - p(background), the better ranking signal")
    ap.add_argument("--nms", type=float, default=0.0,
                    help="per-class 3D IoU suppression threshold; 0 disables")
    ap.add_argument("--min-score", type=float, default=0.05)
    a = ap.parse_args()
    OPT = {"reject_bg": a.reject_bg, "score_mode": a.score_mode,
           "nms": a.nms, "min_score": a.min_score}

    cfg = Config.load(a.config, canon=a.canon)
    ckpt_p = a.ckpt or (cfg.run_dir / cfg.canon / "best.pt")
    if not ckpt_p.exists():
        raise SystemExit(f"no checkpoint at {ckpt_p}")
    ck = torch.load(ckpt_p, map_location=cfg.device, weights_only=False)
    for k in ("canon", "in_ch", "width", "num_classes", "dropout"):
        if k in ck["cfg"]:
            setattr(cfg, k, ck["cfg"][k])
    model = build(cfg).to(cfg.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_p}   canon={cfg.canon}")

    frames = val_frame_ids(cfg)
    if a.max_frames:
        frames = frames[:a.max_frames]
    print(f"evaluating on {len(frames):,} validation frames\n")

    np.random.seed(cfg.seed)
    all_dets, all_gt = [], []
    n_det = 0
    clu_tot = defaultdict(int)
    clu_hit = defaultdict(int)
    rng_tot = defaultdict(int)
    rng_hit = defaultdict(int)

    for f in tqdm(frames, ncols=78, desc="frames"):
        fid = f"{f:06d}"
        dets, objs, hits, calib = run_frame(fid, cfg, model, cfg.device, OPT)
        for oi, o in enumerate(objs):
            cname = CLASSES[o.class_id]
            for d in difficulties_of(o):
                clu_tot[(cname, d)] += 1
                if hits.get(oi, False):
                    clu_hit[(cname, d)] += 1
            rr = float(np.linalg.norm(o.loc_cam[[0, 2]]))
            band = "0-20" if rr < 20 else "20-40" if rr < 40 else "40+"
            rng_tot[(cname, band)] += 1
            if hits.get(oi, False):
                rng_hit[(cname, band)] += 1
        n_det += len(dets)
        all_dets.append(dets)
        all_gt.append({"boxes": gt_boxes_velo(objs, calib),
                       "cls": np.array([o.class_id for o in objs], dtype=int),
                       "diff": [difficulties_of(o) for o in objs]})

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("  1. CLUSTER RECALL  -- the ceiling on everything below")
    print("=" * 74)
    print(f"  {'class':<12}{'easy':>12}{'moderate':>12}{'hard':>12}{'all':>12}")
    print("  " + "-" * 60)
    for c in CLASSES[1:]:
        row = [c]
        tot_a = hit_a = 0
        for d in ("easy", "moderate", "hard"):
            t, h = clu_tot[(c, d)], clu_hit[(c, d)]
            tot_a += t; hit_a += h
            row.append(f"{100*h/t:.1f}%" if t else "-")
        row.append(f"{100*hit_a/tot_a:.1f}%" if tot_a else "-")
        print(f"  {row[0]:<12}{row[1]:>12}{row[2]:>12}{row[3]:>12}{row[4]:>12}")
    print("\n  Objects missed here never reach the network. No amount of")
    print("  classifier accuracy recovers them.")

    print("\n  by range:")
    print(f"  {'class':<12}{'0-20 m':>12}{'20-40 m':>12}{'40+ m':>12}")
    print("  " + "-" * 48)
    for c in CLASSES[1:]:
        cells = []
        for b in ("0-20", "20-40", "40+"):
            t, h = rng_tot[(c, b)], rng_hit[(c, b)]
            cells.append(f"{100*h/t:.1f}%" if t else "-")
        print(f"  {c:<12}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("  2. 3D AP  -- end to end, KITTI protocol")
    print("=" * 74)
    print(f"  {'class':<12}{'IoU':>6}{'easy (n)':>16}{'moderate (n)':>16}{'hard (n)':>16}")
    print("  " + "-" * 66)

    for ci, cname in enumerate(CLASSES[1:], start=1):
        thr = IOU_THRESH[cname]
        cells, counts = [], []
        for diff in ("easy", "moderate", "hard"):
            # KITTI counts a detection as correct only against GT of the same
            # difficulty, and each GT may be matched at most once. Detections
            # are consumed highest-score-first, which is what makes the
            # precision-recall curve meaningful.
            n_gt = 0
            scored = []          # (score, is_true_positive)
            for fi in range(len(all_dets)):
                g = all_gt[fi]
                idx = [j for j in range(len(g["cls"]))
                       if g["cls"][j] == ci and diff in g["diff"][j]]
                n_gt += len(idx)
                used = set()
                cand = sorted([d for d in all_dets[fi] if d["cls"] == ci],
                              key=lambda d: -d["score"])
                for d in cand:
                    best, bj = 0.0, -1
                    for j in idx:
                        if j in used:
                            continue
                        iou = box_iou_3d(d["box"], g["boxes"][j])
                        if iou > best:
                            best, bj = iou, j
                    if best >= thr and bj >= 0:
                        used.add(bj)
                        scored.append((d["score"], 1))
                    else:
                        scored.append((d["score"], 0))
            if n_gt == 0 or not scored:
                cells.append("-")
                continue
            scored.sort(key=lambda t: -t[0])
            tp = np.cumsum([t[1] for t in scored])
            fp = np.cumsum([1 - t[1] for t in scored])
            rec = tp / n_gt
            prec = tp / np.maximum(tp + fp, 1e-9)
            cells.append(f"{100*average_precision(rec, prec):.2f}")
            counts.append(n_gt)
        print(f"  {cname:<12}{thr:>6.1f}"
              f"{cells[0]+' ('+str(counts[0])+')':>16}"
              f"{cells[1]+' ('+str(counts[1])+')':>16}"
              f"{cells[2]+' ('+str(counts[2])+')':>16}")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 74)
    print("  3. AP vs IoU THRESHOLD  -- is the loss detection, or box geometry?")
    print("=" * 74)
    print("  A steep fall from 0.3 to 0.7 means objects ARE being found and")
    print("  classified, but the predicted boxes are not tight enough. A flat")
    print("  low line means they are not being found at all.")
    print()
    sweep = [0.3, 0.5, 0.7]
    print(f"  {'class':<12}" + "".join(f"{'IoU '+str(t):>12}" for t in sweep))
    print("  " + "-" * 48)
    for ci, cname in enumerate(CLASSES[1:], start=1):
        cells = []
        for t in sweep:
            n_gt, scored = 0, []
            for fi in range(len(all_dets)):
                g = all_gt[fi]
                idx = [j for j in range(len(g["cls"]))
                       if g["cls"][j] == ci and "moderate" in g["diff"][j]]
                n_gt += len(idx)
                used = set()
                for d in sorted([d for d in all_dets[fi] if d["cls"] == ci],
                                key=lambda d: -d["score"]):
                    best, bj = 0.0, -1
                    for j in idx:
                        if j in used:
                            continue
                        v = box_iou_3d(d["box"], g["boxes"][j])
                        if v > best:
                            best, bj = v, j
                    if best >= t and bj >= 0:
                        used.add(bj); scored.append((d["score"], 1))
                    else:
                        scored.append((d["score"], 0))
            if n_gt == 0 or not scored:
                cells.append("-"); continue
            scored.sort(key=lambda z: -z[0])
            tp = np.cumsum([z[1] for z in scored])
            fp = np.cumsum([1 - z[1] for z in scored])
            cells.append(f"{100*average_precision(tp/n_gt, tp/np.maximum(tp+fp,1e-9)):.2f}")
        print(f"  {cname:<12}" + "".join(f"{c:>12}" for c in cells))
    print()
    print("  (moderate difficulty)")

    print()
    print("  AP in percent, KITTI 3D protocol. Car is scored at IoU 0.7,")
    print("  Pedestrian and Cyclist at 0.5, matching the official benchmark.")
    print("  These are end-to-end numbers: an object lost by clustering counts")
    print("  as a miss here, unlike the training F1.")


if __name__ == "__main__":
    main()
