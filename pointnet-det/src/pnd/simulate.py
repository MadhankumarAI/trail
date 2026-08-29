"""
Run the trained detector over a continuous raw-KITTI drive and export the result
for the browser visualiser.

    python -m pnd.simulate --ckpt ../best.pt --drive ../data/raw --out ../web/sim.json

Why the raw drive and not the detection split: KITTI's object-detection frames
are shuffled and mutually independent, so playing them back jumps around the
world. `2011_09_26_drive_0001` is 108 *consecutive* sweeps down one street at
10 Hz, which is what a perception system actually sees.

ONE HONEST CONSTRAINT
---------------------
The model was trained only on clusters inside the camera frustum, because that
is the only region KITTI annotates. Classifying the full 360 degrees would be
running it out of distribution and quietly reporting the results as if they were
trustworthy. So: ground removal and the map run over everything, but clustering
and classification run only within the trained field of view. Points outside it
are exported as `unscored` and drawn greyed out, which is both honest and makes
the valid region obvious.

Point classes exported, mapped onto the three categories PS 26053 asks for:

    0 unscored        outside the trained field of view
    1 terrain         ground surface (drivable)
    2 static          non-ground cluster the model calls background: walls,
                      poles, vegetation, buildings
    3 Car             \\
    4 Pedestrian       > dynamic-capable classes
    5 Cyclist         /
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import torch

from .boxes import decode_heading
from .cluster import cluster_points
from .config import Config
from .dataset import ANCHORS, _rot_z
from .ground import remove_ground
from .kitti import read_velodyne
from .model import build

# KITTI's left colour camera spans roughly +/-40 degrees. Matching it keeps the
# model inside the distribution it was trained on.
FOV_DEG = 40.0

CLS_UNSCORED, CLS_TERRAIN, CLS_STATIC = 0, 1, 2
CLS_OFFSET = 2          # Car(1) -> 3, Pedestrian(2) -> 4, Cyclist(3) -> 5


@torch.no_grad()
def process(pts, cfg, model, device):
    """One sweep. Returns (per-point class, boxes, timings, counts)."""
    t = {}
    N = len(pts)
    xy = pts[:, :2]
    rng = np.linalg.norm(xy, axis=1)
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    in_fov = (np.abs(az) <= FOV_DEG) & (rng < cfg.max_range) & (rng > 1.0)

    t0 = time.perf_counter()
    is_ground, agl, terr = remove_ground(pts[:, :3], thresh=cfg.ground_thresh)
    t["ground"] = (time.perf_counter() - t0) * 1000

    cls = np.full(N, CLS_UNSCORED, np.uint8)
    cls[is_ground] = CLS_TERRAIN                    # terrain everywhere

    work = in_fov & ~is_ground
    idx_work = np.flatnonzero(work)
    boxes = []
    n_clusters = 0

    if len(idx_work) >= 50:
        op = pts[idx_work]
        oa = agl[idx_work]

        t0 = time.perf_counter()
        lab = cluster_points(op[:, :3], voxel=cfg.cluster_voxel,
                             min_points=cfg.min_cluster_pts,
                             max_points=cfg.max_cluster_pts)
        t["cluster"] = (time.perf_counter() - t0) * 1000
        n_clusters = int(lab.max()) + 1

        # everything clustered starts as a static obstacle; the network can
        # promote it to a dynamic class
        cls[idx_work[lab >= 0]] = CLS_STATIC

        if n_clusters > 0:
            P, members = [], []
            for k in range(n_clusters):
                m = lab == k
                ii = np.flatnonzero(m)
                if len(ii) < cfg.min_cluster_pts:
                    continue
                sel = (np.random.choice(ii, cfg.n_points, replace=False)
                       if len(ii) >= cfg.n_points
                       else np.random.choice(ii, cfg.n_points, replace=True))
                q = op[sel]
                P.append(np.column_stack([q[:, 0], q[:, 1], q[:, 2],
                                          q[:, 3], oa[sel]]))
                members.append(ii)

            if P:
                t0 = time.perf_counter()
                arr = np.stack(P).astype(np.float64)
                B, PN = arr.shape[0], arr.shape[1]
                xyz = arr[:, :, :3]
                inten, aglv = arr[:, :, 3], arr[:, :, 4]
                rraw = np.linalg.norm(xyz, axis=2)

                from .bench_canon import pca2_batch
                yc = np.zeros(B)
                pca2_batch(np.ascontiguousarray(xyz.reshape(-1, 3)),
                           (np.arange(B + 1) * PN).astype(np.int64), yc)
                tc = xyz.mean(1)
                Rc = _rot_z(-yc)
                xc = np.einsum("bij,bpj->bpi", Rc, xyz - tc[:, None, :])
                scale = np.maximum(np.linalg.norm(xc, axis=2).max(1), 1e-6)
                xc /= scale[:, None, None]

                feats = np.concatenate([
                    xc, (aglv / 3.0)[:, :, None],
                    (rraw / cfg.max_range)[:, :, None],
                    inten[:, :, None]], axis=2).transpose(0, 2, 1)
                x = torch.from_numpy(feats).float().to(device)
                with torch.autocast(device,
                                    dtype=getattr(torch, cfg.amp_dtype)
                                    if cfg.amp_dtype else torch.float32,
                                    enabled=cfg.amp_dtype is not None):
                    out = model(x)

                prob = out["logits"].float().softmax(1).cpu().numpy()
                dc = out["center"].float().cpu().numpy()
                sl = out["size_log"].float().cpu().numpy()
                hb = out["head_bin"].float().argmax(1)
                hr = out["head_res"].float().gather(1, hb.unsqueeze(1)).squeeze(1)
                yaw_p = decode_heading(hb.float(), hr).cpu().numpy()
                yaw_off = np.arctan2(Rc[:, 1, 0], Rc[:, 0, 0])
                t["infer"] = (time.perf_counter() - t0) * 1000

                for i in range(B):
                    top = int(prob[i].argmax())
                    if top == 0:
                        continue                      # stays a static obstacle
                    score = float(1.0 - prob[i, 0])    # foreground mass ranks best
                    if score < 0.5:
                        continue
                    cls[idx_work[members[i]]] = top + CLS_OFFSET
                    ctr = Rc[i].T @ (dc[i] * scale[i]) + tc[i]
                    dims = np.exp(sl[i]) * ANCHORS[top]
                    boxes.append({
                        "c": top, "s": round(score, 3),
                        "b": [round(float(v), 2) for v in
                              [ctr[0], ctr[1], ctr[2], dims[0], dims[1], dims[2],
                               float(yaw_p[i] - yaw_off[i])]]})

    t.setdefault("cluster", 0.0)
    t.setdefault("infer", 0.0)
    return cls, boxes, t, n_clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--drive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--points", type=int, default=5000,
                    help="points exported per frame (display only)")
    a = ap.parse_args()

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
    print(f"model      {a.ckpt}  canon={cfg.canon}  "
          f"F1={ck.get('metrics', {}).get('f1_fg', 0):.4f}")

    scans = sorted(a.drive.rglob("velodyne_points/data/*.bin"))
    if a.max_frames:
        scans = scans[:a.max_frames]
    if not scans:
        raise SystemExit(f"no scans under {a.drive}")
    print(f"drive      {len(scans)} consecutive sweeps")

    np.random.seed(0)
    frames, blob = [], bytearray()
    tot = {"ground": 0.0, "cluster": 0.0, "infer": 0.0}

    for n, sp in enumerate(scans):
        pts = read_velodyne(sp)
        cls, boxes, t, ncl = process(pts, cfg, model, cfg.device)
        for k in tot:
            tot[k] += t[k]

        # subsample for display, keeping every dynamic-class point: they are
        # what the demo is about and there are few of them
        dyn = np.flatnonzero(cls >= 3)
        rest = np.flatnonzero(cls < 3)
        budget = max(a.points - len(dyn), 0)
        if len(rest) > budget:
            rest = np.random.choice(rest, budget, replace=False)
        keep = np.concatenate([dyn, rest])

        q = np.clip(np.round(pts[keep, :3] * 50), -32768, 32767).astype(np.int16)
        blob += q.tobytes() + cls[keep].astype(np.uint8).tobytes()

        frames.append({
            "n": int(len(keep)), "raw": int(len(pts)),
            "cl": ncl, "d": boxes,
            "t": {k: round(v, 1) for k, v in t.items()},
        })
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(scans)}")

    payload = {
        "frames": frames,
        "classes": ["unscored", "terrain", "static", "Car", "Pedestrian", "Cyclist"],
        "quant": 50,          # int16 units per metre
        "fov": FOV_DEG,
        "model": {"canon": cfg.canon,
                  "f1": round(float(ck.get("metrics", {}).get("f1_fg", 0)), 4),
                  "params": sum(p.numel() for p in model.parameters()),
                  "ctr_err": round(float(ck.get("metrics", {}).get("ctr_err", 0)), 2),
                  "yaw_err": round(float(ck.get("metrics", {}).get("yaw_err", 0)), 1)},
        "blob": base64.b64encode(bytes(blob)).decode("ascii"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload))

    nf = len(scans)
    print(f"\nmean per frame:  ground {tot['ground']/nf:.1f} ms   "
          f"cluster {tot['cluster']/nf:.1f} ms   infer {tot['infer']/nf:.1f} ms")
    print(f"                 total {(sum(tot.values()))/nf:.1f} ms  "
          f"= {1000*nf/sum(tot.values()):.1f} FPS")
    print(f"detections       {sum(len(f['d']) for f in frames)} over {nf} frames")
    print(f"wrote            {a.out}  {a.out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
