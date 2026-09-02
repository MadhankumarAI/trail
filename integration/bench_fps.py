"""End-to-end timing for the 2.5D perception stack.

The problem statement asks for FPS. This measures it on real sweeps, stage by
stage, so the number can be defended rather than quoted.

WHAT IS AND IS NOT COUNTED

  counted      ground segmentation, quantisation into the 2.5D grid,
               foveation, drivability, temporal accumulation, and -- when a
               checkpoint is given -- clustering and the detector forward pass
  reported     file reads, separately. A deployed sensor hands you the points
               in memory; charging the benchmark for reading them off a disk
               would flatter or penalise it depending on the disk, and either
               way it measures the wrong thing.
  excluded     the first `--warmup` frames. numba compiles on first call and
               CUDA has its own start-up; including those turns a 20 ms stage
               into a 3 s one and makes the mean meaningless.

Timings are per frame. Median and p95 are both reported because a planner
lives on the worst case, not the average: a stack that runs at 40 FPS median
and 8 FPS at p95 will drop frames exactly when the scene is busiest, which is
when it matters.

GPU work is synchronised before the clock is read. Without that, CUDA calls
return immediately and the detector appears to take almost no time at all.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import grid25 as g
import accumulate as A
import terrain_cells as tc


class Stage:
    """A named timer that keeps every sample, not a running mean."""

    def __init__(self, name):
        self.name = name
        self.ms = []
        self._t = None

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms.append((time.perf_counter() - self._t) * 1000.0)
        return False

    def stats(self, skip=0):
        v = np.array(self.ms[skip:])
        if not len(v):
            return None
        return {"median": float(np.median(v)), "mean": float(v.mean()),
                "p95": float(np.percentile(v, 95)), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--pnd", type=Path, required=True)
    ap.add_argument("--poses", type=Path, default=None)
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="include the detector; omitted = terrain stack only")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(a.pnd))
    from pnd.ground import remove_ground

    torch = sync = None
    model = cfg = process = None
    if a.ckpt:
        import torch as _torch
        torch = _torch
        from pnd.config import Config
        from pnd.model import build
        from pnd.simulate import process
        cfg = Config.load()
        if a.device:
            cfg.device = a.device
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

        def sync():
            if cfg.device == "cuda":
                torch.cuda.synchronize()

    files = sorted(a.cache.glob(f"{a.seq}_*.bin"))[:a.frames + a.warmup]
    if not files:
        raise SystemExit(f"no sweeps under {a.cache}")

    T = (A.load_poses(a.poses, a.calib)
         if (a.poses and a.calib) else None)

    S = {k: Stage(k) for k in ("read", "ground", "quantise", "foveate",
                               "drivability", "accumulate", "detect")}
    total = Stage("total")
    wm = A.WorldMap()
    npts = []

    for n, f in enumerate(files):
        idx = int(f.stem.split("_")[1])

        with S["read"]:
            raw = np.fromfile(f, np.float32).reshape(-1, 4).astype(np.float64)
        npts.append(len(raw))
        pts = raw[:, :3]

        with total:
            with S["ground"]:
                isg, _, _ = remove_ground(pts.astype(np.float32))
            lab = np.where(isg, g.road, g.other)

            if model is not None:
                with S["detect"]:
                    process(raw, cfg, model, cfg.device, None)
                    sync()

            with S["quantise"]:
                cells = g.quantise(pts[:, 0], pts[:, 1], pts[:, 2], lab)

            if T is not None:
                with S["accumulate"]:
                    wm.ingest(pts, lab, T[idx])
                with S["foveate"]:
                    wm.cells_sensor_frame(T[idx])
                src = tc.coarsen(wm.c)
            else:
                with S["foveate"]:
                    src = tc.coarsen(cells)

            with S["drivability"]:
                tc.cell_drivability(src, (0.0, 0.0),
                                    res=g.res0 * (1 << tc.DRIVE_LEVEL))

    sk = a.warmup
    dev = cfg.device if cfg else "cpu (no detector)"
    print(f"\n{len(files) - sk} frames timed, {sk} discarded as warm-up")
    print(f"device {dev}   {np.mean(npts):,.0f} points per sweep average\n")

    print(f"  {'stage':<16}{'median ms':>12}{'mean ms':>10}{'p95 ms':>10}"
          f"{'% of total':>12}")
    print("  " + "-" * 60)
    tot = total.stats(sk)
    for k in ("ground", "detect", "quantise", "accumulate", "foveate",
              "drivability"):
        s = S[k].stats(sk)
        if not s:
            continue
        print(f"  {k:<16}{s['median']:>12.2f}{s['mean']:>10.2f}"
              f"{s['p95']:>10.2f}{100*s['median']/tot['median']:>11.1f}%")
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<16}{tot['median']:>12.2f}{tot['mean']:>10.2f}"
          f"{tot['p95']:>10.2f}")

    rd = S["read"].stats(sk)
    print(f"\n  (file read {rd['median']:.2f} ms median, not counted -- a "
          f"sensor hands\n   you the points in memory)")

    print(f"\n  FPS   median {1000/tot['median']:6.1f}"
          f"    worst case (p95) {1000/tot['p95']:6.1f}")
    hz = 10.0
    head = 1000.0 / hz
    print(f"  a {hz:.0f} Hz lidar leaves {head:.0f} ms per sweep: "
          f"{'within budget' if tot['p95'] < head else 'OVER BUDGET'} "
          f"at p95 ({tot['p95']:.1f} ms)")


if __name__ == "__main__":
    main()
