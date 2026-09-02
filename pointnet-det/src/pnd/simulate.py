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
    1-3 terrain       ground surface: drivable / marginal / non-drivable
    4 static          non-ground cluster the model calls background:
                      walls, poles, vegetation, buildings
    5..               one index per trained object class, in kitti.CLASSES
                      order, so the exported set follows the checkpoint
                      rather than being written out here
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import io

import numpy as np
import torch

from .boxes import decode_heading
from .cluster import cluster_points
from .config import Config
from .dataset import ANCHORS, _rot_z
from .ground import remove_ground
from .kitti import read_velodyne
from .terrain import (DRIVABLE, MARGINAL, NON_DRIVABLE, TerrainTracker,
                      analyse, drivability_score, sector_features)
from .model import build

# KITTI's left colour camera spans roughly +/-40 degrees. Matching it keeps the
# model inside the distribution it was trained on.
FOV_DEG = 40.0

def load_raw_calib(drive: Path):
    """Raw KITTI splits calibration across two files, unlike the detection set.

    calib_velo_to_cam.txt gives R (3x3) and T (3x1); calib_cam_to_cam.txt gives
    R_rect_00 and P_rect_02. Assemble them into the same projection chain the
    detection benchmark ships pre-joined:  uv ~ P_rect_02 . R_rect . [R|T] . p
    """
    d = None
    for c in drive.rglob("calib_cam_to_cam.txt"):
        d = c.parent
        break
    if d is None:
        return None

    def vals(path, key):
        for line in Path(path).read_text().splitlines():
            if line.startswith(key + ":"):
                return np.array([float(v) for v in line.split(":", 1)[1].split()])
        return None

    R = vals(d / "calib_velo_to_cam.txt", "R").reshape(3, 3)
    T = vals(d / "calib_velo_to_cam.txt", "T").reshape(3, 1)
    R_rect = vals(d / "calib_cam_to_cam.txt", "R_rect_00").reshape(3, 3)
    P2 = vals(d / "calib_cam_to_cam.txt", "P_rect_02").reshape(3, 4)
    size = vals(d / "calib_cam_to_cam.txt", "S_rect_02")
    return {"V2C": np.hstack([R, T]), "R0": R_rect, "P2": P2,
            "W": int(size[0]), "H": int(size[1])}


def project(pts, calib):
    """(N,3) velodyne -> (N,2) pixels and a mask of what the camera can see."""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (calib["R0"] @ (calib["V2C"] @ h.T)).T
    ph = np.hstack([cam, np.ones((len(cam), 1))])
    uv = (calib["P2"] @ ph.T).T
    z = uv[:, 2:3]
    uv = uv[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)
    vis = ((cam[:, 2] > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < calib["W"])
           & (uv[:, 1] >= 0) & (uv[:, 1] < calib["H"]))
    return uv, vis


CLS_UNSCORED = 0
CLS_DRIVABLE, CLS_MARGINAL, CLS_NONDRIV = 1, 2, 3
CLS_STATIC = 4
CLS_OFFSET = 4          # object classes follow CLS_STATIC in kitti.CLASSES
                        # order, so the exported set grows with the class list


def display_classes():
    """Names for the exported point classes, derived rather than hardcoded.

    This list used to end at Cyclist. A checkpoint trained with more classes
    then emits indices with no name against them, and the viewer's legend and
    colour table run off the end -- which shows up as objects drawn in the
    wrong colour, not as an error.
    """
    from .kitti import CLASSES
    return (["unscored", "drivable", "marginal", "non-drivable", "static"]
            + list(CLASSES[1:]))

# terrain.py's DRIVABLE/MARGINAL/NON_DRIVABLE are 0/1/2
_TERR = {DRIVABLE: CLS_DRIVABLE, MARGINAL: CLS_MARGINAL,
         NON_DRIVABLE: CLS_NONDRIV}


@torch.no_grad()
def process(pts, cfg, model, device, tracker=None):
    """One sweep. Returns (per-point class, boxes, timings, counts)."""
    t = {}
    N = len(pts)
    xy = pts[:, :2]
    rng = np.linalg.norm(xy, axis=1)
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    in_fov = (np.abs(az) <= FOV_DEG) & (rng < cfg.max_range) & (rng > 1.0)

    t0 = time.perf_counter()
    is_ground, agl, stats = remove_ground(pts[:, :3], thresh=cfg.ground_thresh)
    t["ground"] = (time.perf_counter() - t0) * 1000

    # Drivability over the whole sweep, not just the trained field of view:
    # it is geometry, so unlike the classifier it is valid everywhere.
    t0 = time.perf_counter()
    if tracker is not None:
        feat = sector_features(stats)
        sc = drivability_score(feat, n_radial=stats["n_radial"],
                               n_azimuth=stats["n_azimuth"])
        sc = tracker.update(sc)
        terr = analyse(pts[:, :3], is_ground, stats, score=sc)
    else:
        terr = analyse(pts[:, :3], is_ground, stats)
    t["terrain"] = (time.perf_counter() - t0) * 1000

    cls = np.full(N, CLS_UNSCORED, np.uint8)
    pcls = terr["point_cls"]
    for k, v in _TERR.items():
        cls[is_ground & (pcls == k)] = v

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
                    # Wrap to (-pi, pi]. The heading is a decoded bin plus a
                    # residual minus the canonicalisation offset, none of which
                    # is wrapped, so raw values reached 455 and -102 degrees.
                    # Drawing is unaffected because sin and cos do not care,
                    # but anything that compares, averages or thresholds a
                    # heading does, and a tracker would be the first to break.
                    yaw = float(yaw_p[i] - yaw_off[i])
                    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                    boxes.append({
                        "c": top, "s": round(score, 3),
                        "b": [round(float(v), 2) for v in
                              [ctr[0], ctr[1], ctr[2], dims[0], dims[1], dims[2],
                               yaw]]})

    t.setdefault("cluster", 0.0)
    t.setdefault("infer", 0.0)
    # Sector-level terrain, so the viewer can draw the decision surface as
    # filled wedges. Drawing 5,000 sampled points out of 122,460 makes a
    # perfectly coherent surface look speckled - the sparseness is the sampling,
    # not the classifier.
    sec_cls = terr["sector_cls"].astype(np.uint8)
    sec_h = np.clip(np.round(terr["feat"]["h"] * 100), -32768, 32767).astype(np.int16)
    return cls, boxes, t, n_clusters, sec_cls, sec_h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--drive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--points", type=int, default=5000,
                    help="points exported per frame (display only)")
    ap.add_argument("--image-width", type=int, default=512,
                    help="camera frames are downscaled to this before embedding")
    ap.add_argument("--image-quality", type=int, default=55)
    ap.add_argument("--fuse-alpha", type=float, default=0.4,
                    help="temporal fusion weight for drivability; lower = steadier")
    ap.add_argument("--no-fuse", action="store_true",
                    help="disable temporal fusion (per-frame, chatters)")
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

    calib = load_raw_calib(a.drive)
    print(f"calibration {'loaded' if calib else 'NOT FOUND - overlay disabled'}")

    scans = sorted(a.drive.rglob("velodyne_points/data/*.bin"))
    if a.max_frames:
        scans = scans[:a.max_frames]
    if not scans:
        raise SystemExit(f"no scans under {a.drive}")
    print(f"drive      {len(scans)} consecutive sweeps")

    np.random.seed(0)
    tracker = None if a.no_fuse else TerrainTracker(alpha=a.fuse_alpha)
    print(f"terrain fusion   {'off' if tracker is None else f'EMA alpha={a.fuse_alpha}'}")
    frames, blob = [], bytearray()
    # Three separate runs -- all u, then all v, then all classes -- rather than
    # interleaving 5 bytes per point. Interleaved, a frame with an odd point
    # count leaves the next frame starting on an odd byte, and Int16Array
    # refuses a byte offset that is not a multiple of 2. Grouping by field means
    # each typed array is built once over its own contiguous run and per-frame
    # slices are taken in elements, where alignment cannot go wrong.
    cam_u, cam_v, cam_c = [], [], []
    sec_c_all, sec_h_all = [], []
    images = []
    IMG_W = a.image_width
    tot = {"ground": 0.0, "terrain": 0.0, "cluster": 0.0, "infer": 0.0}

    for n, sp in enumerate(scans):
        pts = read_velodyne(sp)
        cls, boxes, t, ncl, sec_cls, sec_h = process(
            pts, cfg, model, cfg.device, tracker)
        sec_c_all.append(sec_cls); sec_h_all.append(sec_h)
        for k in tot:
            tot[k] += t[k]

        # subsample for display, keeping every dynamic-class point: they are
        # what the demo is about and there are few of them
        dyn = np.flatnonzero(cls >= CLS_OFFSET + 1)
        rest = np.flatnonzero(cls < CLS_OFFSET + 1)
        budget = max(a.points - len(dyn), 0)
        if len(rest) > budget:
            rest = np.random.choice(rest, budget, replace=False)
        keep = np.concatenate([dyn, rest])

        q = np.clip(np.round(pts[keep, :3] * 50), -32768, 32767).astype(np.int16)
        blob += q.tobytes() + cls[keep].astype(np.uint8).tobytes()

        # ---- camera overlay -------------------------------------------- #
        # Project the same displayed points into the image and keep only what
        # the camera can actually see, so the overlay is a genuine reprojection
        # of the prediction rather than a redrawn approximation of it.
        n_cam = 0
        if calib is not None:
            uv, vis = project(pts[keep, :3].astype(np.float64), calib)
            sc = IMG_W / calib["W"]
            cu = np.clip(np.round(uv[vis, 0] * sc), 0, 32767).astype(np.int16)
            cv = np.clip(np.round(uv[vis, 1] * sc), 0, 32767).astype(np.int16)
            cc = cls[keep][vis].astype(np.uint8)
            cam_u.append(cu); cam_v.append(cv); cam_c.append(cc)
            n_cam = int(vis.sum())

            img_p = sp.parent.parent.parent / "image_02" / "data" / f"{sp.stem}.png"
            if img_p.exists():
                from PIL import Image
                im = Image.open(img_p).convert("RGB")
                im = im.resize((IMG_W, int(round(im.height * sc))), Image.BILINEAR)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=a.image_quality, optimize=True)
                images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            else:
                images.append("")

        frames.append({
            "n": int(len(keep)), "raw": int(len(pts)),
            "cl": ncl, "d": boxes, "nc": n_cam,
            "t": {k: round(v, 1) for k, v in t.items()},
        })
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(scans)}")

    payload = {
        "frames": frames,
        "classes": display_classes(),
        "quant": 50,          # int16 units per metre
        "grid": {"nr": 24, "na": 72, "R": 70.0,
                 "cls": base64.b64encode(
                     np.concatenate(sec_c_all).tobytes()).decode("ascii"),
                 "h": base64.b64encode(
                     np.concatenate(sec_h_all).tobytes()).decode("ascii")},
        "fov": FOV_DEG,
        "model": {"canon": cfg.canon,
                  "f1": round(float(ck.get("metrics", {}).get("f1_fg", 0)), 4),
                  "params": sum(p.numel() for p in model.parameters()),
                  "ctr_err": round(float(ck.get("metrics", {}).get("ctr_err", 0)), 2),
                  "yaw_err": round(float(ck.get("metrics", {}).get("yaw_err", 0)), 1)},
        "blob": base64.b64encode(bytes(blob)).decode("ascii"),
        "cam": {
            "blob": base64.b64encode(
                (np.concatenate(cam_u).tobytes() if cam_u else b"")
                + (np.concatenate(cam_v).tobytes() if cam_v else b"")
                + (np.concatenate(cam_c).tobytes() if cam_c else b"")
            ).decode("ascii"),
            "total": int(sum(len(x) for x in cam_u)),
            "images": images,
            "w": IMG_W,
            "h": int(round(calib["H"] * IMG_W / calib["W"])) if calib else 0,
        } if calib is not None else None,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload))

    nf = len(scans)
    print(f"\nmean per frame:  ground {tot['ground']/nf:.1f} ms   "
          f"terrain {tot['terrain']/nf:.2f} ms   "
          f"cluster {tot['cluster']/nf:.1f} ms   infer {tot['infer']/nf:.1f} ms")
    print(f"                 total {(sum(tot.values()))/nf:.1f} ms  "
          f"= {1000*nf/sum(tot.values()):.1f} FPS")
    print(f"detections       {sum(len(f['d']) for f in frames)} over {nf} frames")
    if images:
        mb = sum(len(i) for i in images) / 1e6
        print(f"camera overlay   {len(images)} frames, {mb:.2f} MB of JPEG")
    print(f"wrote            {a.out}  {a.out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
