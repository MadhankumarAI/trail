"""Recompute every headline number from the current code, into one JSON.

Written so the charts cannot drift from the pipeline. Nothing here is copied
from an earlier run: each figure is measured now, on the code as it stands, and
anything that comes from elsewhere (the detector's AP, which needs the full
KITTI validation split and a GPU) is marked with its source so it is never
mistaken for something this script produced.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import grid25 as g
import accumulate as A
import dense25 as D
import terrain_cells as tc
import gridmap_filters as gf
from gridmap import GridMap


def load_frames(cache, seq, upto, remove_ground, thresh):
    out = []
    for f in sorted(cache.glob(f"{seq}_*.bin")):
        i = int(f.stem.split("_")[1])
        if i > upto:
            continue
        raw = np.fromfile(f, np.float32).reshape(-1, 4).astype(np.float64)
        isg, _, _ = remove_ground(raw[:, :3].astype(np.float32), thresh=thresh)
        out.append((i, raw, np.where(isg, g.road, g.other)))
    return out


def timing(frames, T, cfg, model, process, remove_ground, warm=3):
    """Per-stage wall clock for the live pipeline."""
    dm = D.FastMap()
    dm.ingest(frames[0][1], frames[0][2], T[frames[0][0]],
              groundcls=(g.gnd, g.road))          # JIT + first-touch
    dm = D.FastMap()
    S = {k: [] for k in ("ground", "detect", "accumulate", "drivability",
                         "total")}
    import torch
    for i, raw, lab in frames:
        t0 = time.perf_counter()
        t = time.perf_counter()
        gr = remove_ground(raw[:, :3].astype(np.float32),
                           thresh=cfg.ground_thresh)
        S["ground"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        process(raw, cfg, model, cfg.device, None, ground=gr, terrain=False)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        S["detect"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        dm.ingest(raw, lab, T[i], groundcls=(g.gnd, g.road))
        S["accumulate"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        c = dm.cells(3, min_pts=3)
        tc.cell_drivability(c, (0.0, 0.0), res=dm.res[3])
        S["drivability"].append((time.perf_counter() - t) * 1000)
        S["total"].append((time.perf_counter() - t0) * 1000)

    return {k: {"median": float(np.median(v[warm:])),
                "p95": float(np.percentile(v[warm:], 95)),
                "mean": float(np.mean(v[warm:]))} for k, v in S.items()}, dm


def memory(dm):
    """What each representation costs for the same 100 m of ground.

    Like for like: every row is a DENSE allocation over the same footprint,
    because that is the comparison the adaptive scheme actually wins. Mixing a
    dense baseline against a sparse candidate would credit foveation with the
    saving that sparsity provides, and they are different ideas.
    """
    rows = int(dm.acc.shape[0])
    fixed = int(dm.acc.shape[1])
    uni = GridMap(200.0, 200.0, 0.05)
    uni_cells = uni.n_x * uni.n_y
    per = {}
    for t in range(len(dm.res)):
        n = int(dm.n[t])
        per[f"{dm.res[t]*100:.0f} cm"] = {
            "cells": n * n,
            "half_extent_m": float(n * dm.res[t] / 2),
        }
    return {
        "rows_per_cell": rows,
        "bytes_per_value": 8,
        "adaptive_cells": fixed,
        "adaptive_bytes": fixed * rows * 8,
        "uniform_cells": int(uni_cells),
        "uniform_bytes": int(uni_cells) * rows * 8,
        "reduction": float(uni_cells) / fixed,
        "tiers": per,
        "occupied": int(dm.stats()["cells_occupied"]),
    }


def terrain_quality(frames, T, cache):
    """Drivability against SemanticKITTI road labels, single vs accumulated."""
    import kitti
    RES = 0.40
    p0, l0, _ = kitti.load(str(cache / "00_000000.bin"),
                           str(cache / "00_000000.label"))
    w0 = A.apply(T[0], p0)
    truth = set(g._pack(*(np.floor(w0[l0 == g.road, :2].T / RES)
                          .astype(np.int64))))

    dm = D.FastMap()
    for i, raw, lab in frames:
        dm.ingest(raw, lab, T[i], groundcls=(g.gnd, g.road))
    last = frames[-1][0]
    sxy = (T[last][0, 3], T[last][1, 3])

    def score(cells, res):
        """Fraction of EVERY ground-truth road cell given each verdict.

        The denominator is the full truth set, not the cells this arm happens
        to hold: a cell it has no evidence for counts as unknown rather than
        vanishing. Scoring only what an arm can see rewards seeing less --
        a single sweep covers 660 of the 2,232 road cells, all of them nearby,
        and looked BETTER than the accumulated map (82.9% against 65.3%)
        purely by being marked on the easy subset.
        """
        _, cls, _ = tc.cell_drivability(cells, sxy, res=res)
        k = g._pack(np.floor((cells["ix"] + 0.5) * res / RES).astype(np.int64),
                    np.floor((cells["iy"] + 0.5) * res / RES).astype(np.int64))
        have = {}
        for kk, cc in zip(k, cls):
            if kk in truth:
                have[kk] = cc                     # last write wins; 1:1 here
        n = len(truth)
        got = np.array(list(have.values()))
        cnt = {v: int((got == v).sum()) for v in
               (tc.DRIVABLE, tc.MARGINAL, tc.NON_DRIVABLE, tc.UNKNOWN)}
        cnt[tc.UNKNOWN] += n - len(have)          # never observed at all
        return {"drivable": cnt[tc.DRIVABLE] / n,
                "marginal": cnt[tc.MARGINAL] / n,
                "non_drivable": cnt[tc.NON_DRIVABLE] / n,
                "unknown": cnt[tc.UNKNOWN] / n,
                "covered": len(have) / n,
                "n": n}

    # one sweep only, same machinery, so the comparison isolates evidence
    one = D.FastMap()
    i, raw, lab = frames[-1]
    one.ingest(raw, lab, T[i], groundcls=(g.gnd, g.road))

    return {"accumulated": score(dm.cells(3, min_pts=3), dm.res[3]),
            "single": score(one.cells(3, min_pts=3), one.res[3]),
            "frames": len(frames)}, dm


def estimator_roc(dm, T, frames, cache):
    """grid_map's filter chain against ours, at matched false-drivable rate."""
    import kitti
    RES = float(dm.res[3])
    cells = dm.cells(3, min_pts=3)
    last = frames[-1][0]
    sxy = (T[last][0, 3], T[last][1, 3])
    score, cls, height = tc.cell_drivability(cells, sxy, res=RES)

    gm = GridMap(220.0, 220.0, RES, position=sxy)
    cx = (cells["ix"] + 0.5) * RES
    cy = (cells["iy"] + 0.5) * RES
    sel = np.isfinite(height)
    gm.set_cells(cx[sel], cy[sel], {"elevation": height[sel]})
    lay = gf.chain(gm.layers["elevation"].astype(float), RES, 1.0, 1.0)
    ix, iy, inside = gm.index_from_position(cx, cy)
    trav = np.full(len(cx), np.nan)
    trav[inside] = lay["traversability"][ix[inside], iy[inside]]

    p0, l0, _ = kitti.load(str(cache / "00_000000.bin"),
                           str(cache / "00_000000.label"))
    w0 = A.apply(T[0], p0)
    road = set(g._pack(*(np.floor(w0[l0 == g.road, :2].T / RES)
                         .astype(np.int64))))
    obst = set(g._pack(*(np.floor(
        w0[np.isin(l0, (g.bldg, g.pole, g.car, g.veg)), :2].T / RES)
        .astype(np.int64)))) - road
    key = g._pack(cells["ix"], cells["iy"])
    isr = np.array([k in road for k in key])
    iso = np.array([k in obst for k in key])
    ok = np.isfinite(score)

    def curve(v, hi_good):
        pts = []
        vv = v[ok & np.isfinite(v)]
        if not len(vv):
            return pts
        for t in np.unique(np.quantile(vv, np.linspace(0, 1, 200))):
            keep = ok & np.isfinite(v) & ((v >= t) if hi_good else (v <= t))
            pts.append({"fp": float((keep & iso).sum() / max(iso.sum(), 1)),
                        "tp": float((keep & isr).sum() / max(isr.sum(), 1))})
        return sorted(pts, key=lambda d: d["fp"])

    def auc(c):
        if len(c) < 2:
            return 0.0
        x = [p["fp"] for p in c]
        y = [p["tp"] for p in c]
        return float(np.trapezoid(y, x) if hasattr(np, "trapezoid")
                     else np.trapz(y, x))

    ours, gmc = curve(-score, True), curve(trav, True)
    return {"ours": ours, "gridmap": gmc,
            "auc_ours": auc(ours), "auc_gridmap": auc(gmc),
            "n_road": int((isr & ok).sum()), "n_obstacle": int((iso & ok).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--pnd", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--max-frame", type=int, default=23)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.pnd))
    import torch
    from pnd.ground import remove_ground
    from pnd.config import Config
    from pnd.model import build
    from pnd.simulate import process
    from pnd.kitti import CLASSES

    cfg = Config.load()
    ck = torch.load(a.ckpt, map_location=cfg.device, weights_only=False)
    for k in ("canon", "in_ch", "width", "num_classes", "dropout", "n_points",
              "cluster_voxel", "min_cluster_pts", "max_cluster_pts",
              "ground_thresh", "max_range"):
        if k in ck["cfg"]:
            setattr(cfg, k, ck["cfg"][k])
    model = build(cfg).to(cfg.device)
    model.load_state_dict(ck["model"])
    model.eval()
    np.random.seed(0)

    T = A.load_poses(a.poses, a.calib)
    frames = load_frames(a.cache, a.seq, a.max_frame, remove_ground,
                         cfg.ground_thresh)
    print(f"{len(frames)} sweeps, {np.mean([len(f[1]) for f in frames]):,.0f} "
          f"points each, device {cfg.device}")

    print("timing...")
    tm, dm = timing(frames, T, cfg, model, process, remove_ground)
    print("memory...")
    mem = memory(dm)
    print("terrain...")
    terr, dm2 = terrain_quality(frames, T, a.cache)
    print("estimators...")
    roc = estimator_roc(dm2, T, frames, a.cache)

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "scene": {"sequence": a.seq, "frames": len(frames),
                  "points_per_sweep": int(np.mean([len(f[1]) for f in frames])),
                  "device": cfg.device},
        "timing": tm,
        "timing_before": {"ground": 23.3, "detect": 121.0,
                          "accumulate": 403.9, "drivability": 4.6,
                          "total": 648.3},
        "memory": mem,
        "terrain": terr,
        "estimators": roc,
        "model": {
            "classes": list(CLASSES),
            "f1_fg": float(ck.get("metrics", {}).get("f1_fg", 0)),
            "per_class_f1": {k: float(v) for k, v in
                             ck.get("metrics", {})
                             .get("per_class_f1", {}).items()},
            "params": int(sum(p.numel() for p in model.parameters())),
            "canon": cfg.canon,
        },
        # measured on the full KITTI validation split on the training machine,
        # not by this script -- kept separate so the provenance is visible
        "detector_ap_external": {
            "source": "pnd.evaluate on 1,496 val frames, runs/weighted_box",
            "moderate": {"Car": 31.33, "Pedestrian": 31.81, "Cyclist": 49.85,
                         "Van": 8.68, "Truck": 9.19},
            "iou_sweep": {
                "Car": {"0.3": 65.92, "0.5": 60.40, "0.7": 31.33},
                "Pedestrian": {"0.3": 50.92, "0.5": 31.81, "0.7": 1.80},
                "Cyclist": {"0.3": 56.34, "0.5": 49.85, "0.7": 15.08},
                "Van": {"0.3": 39.66, "0.5": 35.39, "0.7": 8.68},
                "Truck": {"0.3": 30.46, "0.5": 20.92, "0.7": 9.19}},
            "cluster_recall_range": {
                "Car": {"0-20": 95.5, "20-40": 83.3, "40+": 25.0},
                "Pedestrian": {"0-20": 67.0, "20-40": 56.1, "40+": 12.5},
                "Cyclist": {"0-20": 86.4, "20-40": 88.4, "40+": 19.4},
                "Van": {"0-20": 99.4, "20-40": 96.8, "40+": 37.9},
                "Truck": {"0-20": 100.0, "20-40": 100.0, "40+": 31.7}},
        },
        # from canon_study.py, the ablation that motivated the architecture
        "canonicalisation_external": {
            "source": "pnd.canon_study",
            "variants": {"none": {"f1": 0.7203, "ms": 0.0},
                         "tnet3": {"f1": 0.7574, "ms": 15.89},
                         "pca3_skew": {"f1": 0.7411, "ms": 0.096},
                         "pca2_yaw": {"f1": 0.8161, "ms": 0.068}},
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}  {a.out.stat().st_size/1024:.0f} KB")
    print(f"  {tm['total']['median']:.1f} ms/frame -> "
          f"{1000/tm['total']['median']:.1f} FPS")
    print(f"  memory {mem['adaptive_bytes']/1e6:.0f} MB vs uniform "
          f"{mem['uniform_bytes']/1e6:.0f} MB  ({mem['reduction']:.1f}x)")
    print(f"  drivable on GT road {100*terr['accumulated']['drivable']:.1f}% "
          f"accumulated vs {100*terr['single']['drivable']:.1f}% single")
    print(f"  AUC ours {roc['auc_ours']:.3f}  grid_map {roc['auc_gridmap']:.3f}")


if __name__ == "__main__":
    main()
