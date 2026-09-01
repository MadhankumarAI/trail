"""Why does a class with good F1 score zero AP?

Van reports F1 0.689 on the validation proposals and 79% cluster recall, yet
0.00 AP even at IoU 0.3. Those cannot all be true: the objects are found, they
are classified, and the boxes are apparently nowhere near them. Something in
between is wrong, and guessing at it from the AP table alone has already cost
one wrong hypothesis.

This dumps the actual boxes. For one class it reports, per frame:

    how many detections carry that class label
    how many ground-truth boxes of that class exist
    the best IoU any detection achieves against any of them
    and for the closest pairs, the geometry side by side

Read-only: loads a checkpoint, runs frames, prints. Trains nothing, writes
nothing.

WHAT THE ANSWER LOOKS LIKE

  best IoU near 0 and centres far apart
      the detections are on different objects entirely -- the class is being
      predicted on the wrong clusters, and the real vans are going out under
      another label

  centres close but IoU still 0
      a geometry bug: dimensions in the wrong order, a yaw convention
      mismatch, or a centre that is bottom-face in one frame and mid-box in
      the other

  no detections at all with that label
      classification never fires at inference, whatever the training F1 said
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .evaluate import box_iou_3d, gt_boxes_velo, run_frame, val_frame_ids
from .kitti import CLASSES
from .model import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--canon", default=None)
    ap.add_argument("--cls", default="Van", help="class name to investigate")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--show", type=int, default=12,
                    help="how many closest pairs to print in full")
    ap.add_argument("--score-mode", choices=["class", "fg"], default="fg")
    ap.add_argument("--nms", type=float, default=0.1)
    ap.add_argument("--min-score", type=float, default=0.05)
    a = ap.parse_args()

    if a.cls not in CLASSES:
        raise SystemExit(f"{a.cls!r} not in {CLASSES}")
    ci = CLASSES.index(a.cls)

    cfg = Config.load(a.config, canon=a.canon)
    ck = torch.load(a.ckpt, map_location=cfg.device, weights_only=False)
    for k in ("canon", "in_ch", "width", "num_classes", "dropout"):
        if k in ck["cfg"]:
            setattr(cfg, k, ck["cfg"][k])
    model = build(cfg).to(cfg.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {a.ckpt}  canon={cfg.canon}  num_classes={cfg.num_classes}")
    print(f"investigating {a.cls!r} (class id {ci})\n")

    OPT = {"reject_bg": False, "score_mode": a.score_mode,
           "nms": a.nms, "min_score": a.min_score}
    frames = val_frame_ids(cfg)[:a.frames]
    np.random.seed(cfg.seed)

    n_det = n_gt = 0
    best_ious, pairs = [], []
    pred_label_of_gt = []          # what the model called each GT of this class

    for f in frames:
        dets, objs, hits, calib = run_frame(f"{f:06d}", cfg, model,
                                            cfg.device, OPT)
        gtb = gt_boxes_velo(objs, calib)
        gidx = [j for j, o in enumerate(objs) if o.class_id == ci]
        mine = [d for d in dets if d["cls"] == ci]
        n_det += len(mine)
        n_gt += len(gidx)

        # for every GT of this class, what is the closest detection of ANY
        # class? that is what reveals a class being emitted under another label
        for j in gidx:
            gb = gtb[j]
            bi, bd = 0.0, None
            for d in dets:
                iou = box_iou_3d(d["box"], gb)
                if iou > bi:
                    bi, bd = iou, d
            if bd is not None:
                pred_label_of_gt.append(CLASSES[bd["cls"]])
                pairs.append((bi, gb, bd["box"], CLASSES[bd["cls"]],
                              bd["score"]))
            else:
                pred_label_of_gt.append("(nothing overlaps)")

        # and the best IoU each of OUR detections manages against its own class
        for d in mine:
            bi = max([box_iou_3d(d["box"], gtb[j]) for j in gidx], default=0.0)
            best_ious.append(bi)

    print(f"over {len(frames)} frames")
    print(f"  ground-truth {a.cls:<12} {n_gt:>6}")
    print(f"  detections labelled {a.cls:<5} {n_det:>6}")
    if n_det == 0:
        print("\n  -> the model never emits this label at inference. The"
              "\n     training F1 is measured on cached proposals, so the two"
              "\n     disagree only if the inference path differs.")
        return

    b = np.array(best_ious)
    print(f"\n  best IoU of a {a.cls} detection against a {a.cls} GT:")
    print(f"    median {np.median(b):.3f}   p90 {np.percentile(b, 90):.3f}   "
          f"max {b.max():.3f}")
    for t in (0.1, 0.3, 0.5, 0.7):
        print(f"    reaching {t:.1f}: {100*(b >= t).mean():5.1f}%")

    from collections import Counter
    print(f"\n  what the model actually calls each {a.cls} ground truth:")
    for name, c in Counter(pred_label_of_gt).most_common():
        print(f"    {name:<20}{c:>6}  {100*c/max(len(pred_label_of_gt),1):5.1f}%")

    pairs.sort(key=lambda p: -p[0])
    print(f"\n  closest pairs -- GT against the best-overlapping detection:")
    print(f"    {'IoU':>6}{'pred':>12}{'score':>7}"
          f"{'d(centre)':>11}{'gt lwh':>22}{'pred lwh':>22}{'d(yaw)':>9}")
    for bi, gb, db, name, sc in pairs[:a.show]:
        dc = float(np.linalg.norm(gb[:3] - db[:3]))
        dy = np.degrees(abs(((gb[6] - db[6]) + np.pi) % (2*np.pi) - np.pi))
        print(f"    {bi:>6.3f}{name:>12}{sc:>7.2f}{dc:>11.2f}"
              f"{f'{gb[3]:.2f},{gb[4]:.2f},{gb[5]:.2f}':>22}"
              f"{f'{db[3]:.2f},{db[4]:.2f},{db[5]:.2f}':>22}{dy:>9.1f}")


if __name__ == "__main__":
    main()
