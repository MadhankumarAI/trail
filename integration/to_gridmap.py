"""Publish the 2.5D terrain layer as grid_map, and measure what it costs.

Two outputs:

  1. a grid_map_msgs/GridMap carrying elevation, variance, observation count,
     traversability and its per-criterion sublayers, and obstacle height --
     ready for RViz's grid_map plugin or any grid_map consumer

  2. the memory comparison the problem statement asks for, against a real
     uniform grid rather than a strawman

ON THE COMPARISON BEING FAIR
----------------------------
An adaptive map beats a dense one for two quite different reasons, and quoting
one number hides that:

    sparsity   most of a 100 m square is never observed at all
    foveation  what IS observed is stored coarser with range

Only the second is the contribution here; the first any hash map would give.
So three figures are reported -- dense uniform, occupied-only uniform, and
occupied-only foveated -- and the foveation gain is measured against the
occupied-only uniform column, which is the honest denominator.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import grid25 as g
import accumulate as A
import terrain_cells as tc
from gridmap import GridMap, LAYERS, cost_to_traversability

# bytes a sparse cell costs us: ix, iy (int32) plus one float32 per layer
SPARSE_BYTES = 8 + 4 * len(LAYERS)


def obstacle_height(cells):
    """How far the tallest return in a cell stands above its own ground.

    Geometric, so unlike the detector's static-obstacle class this is not
    limited to the forward field of view or to things that cluster: a wall
    seen at 60 m behind the vehicle still registers. Cells with no ground
    estimate of their own get nothing rather than a guess.
    """
    ng = cells["ng"]
    base = np.where(ng > 0, cells["gsum"] / np.maximum(ng, 1), np.nan)
    h = cells["zmax"] - base
    return np.where(np.isfinite(h) & (h > 0.30) & (h < 8.0), h, np.nan)


def build(cells, sensor_xy, res, extent, frame_id="map"):
    """A grid_map holding the drivability layer, in the sensor's frame."""
    score, cls, height, term = tc.cell_drivability(cells, sensor_xy, res=res,
                                                   terms=True)
    cx = (cells["ix"] + 0.5) * res
    cy = (cells["iy"] + 0.5) * res

    gm = GridMap(extent, extent, res, position=sensor_xy, frame_id=frame_id)
    ng = np.maximum(cells["ng"], 1)
    ok = np.isfinite(score)

    gm.set_cells(cx[ok], cy[ok], {
        "elevation": height[ok],
        "variance": np.maximum(cells["gsq"][ok] / ng[ok]
                               - (cells["gsum"][ok] / ng[ok]) ** 2, 0.0),
        "n_observations": cells["ng"][ok],
        "traversability": cost_to_traversability(score[ok]),
        "traversability_slope": cost_to_traversability(term["slope"][ok]),
        "traversability_step": cost_to_traversability(term["step"][ok]),
        "traversability_roughness": cost_to_traversability(term["rough"][ok]),
    })
    oh = obstacle_height(cells)
    has = np.isfinite(oh)
    if has.any():
        gm.set_cells(cx[has], cy[has], {"obstacle_height": oh[has]})
    return gm, score, cls


def compare(cells_fine, foveated, res_fine):
    """Dense uniform, occupied-only uniform, occupied-only foveated."""
    n_fov = len(foveated["n"])
    n_occ = len(cells_fine["ix"])
    return {
        "fine_res": res_fine,
        "occupied_uniform_cells": n_occ,
        "foveated_cells": n_fov,
        "occupied_uniform_bytes": n_occ * SPARSE_BYTES,
        "foveated_bytes": n_fov * SPARSE_BYTES,
        "foveation_gain": n_occ / max(n_fov, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--pnd", type=Path, required=True)
    ap.add_argument("--max-frame", type=int, default=23)
    ap.add_argument("--extent", type=float, default=100.0,
                    help="side of the square grid_map, metres")
    ap.add_argument("--msg-res", type=float, default=0.4,
                    help="resolution of the emitted grid_map")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(a.pnd))
    from pnd.ground import remove_ground

    T = A.load_poses(a.poses, a.calib)
    files = sorted(a.cache.glob(f"{a.seq}_*.bin"))
    seq = [(int(f.stem.split("_")[1]), f) for f in files]
    seq = [(i, f) for i, f in seq if i <= a.max_frame]

    wm = A.WorldMap()
    for i, f in seq:
        pts = np.fromfile(f, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        isg, _, _ = remove_ground(pts.astype(np.float32))
        wm.ingest(pts, np.where(isg, g.road, g.other), T[i])
    last = seq[-1][0]
    sxy = (T[last][0, 3], T[last][1, 3])
    print(f"accumulated {len(seq)} frames -> {len(wm.c['ix']):,} fine cells "
          f"at {g.res0:.2f} m\n")

    # the foveated map: variable resolution, tiers following the sensor
    fov = wm.cells_sensor_frame(T[last])
    coarse = tc.coarsen(wm.c)
    gm, score, cls = build(coarse, sxy, g.res0 * (1 << tc.DRIVE_LEVEL),
                           a.extent)

    print("MEMORY, and where the saving actually comes from")
    print("  (two effects: most of the square is never seen, and what IS seen")
    print("   is stored coarser with range. Only the second is foveation.)\n")

    dense = GridMap(a.extent, a.extent, g.res0)
    print(f"  {'representation':<38}{'cells':>14}{'bytes':>14}")
    print("  " + "-" * 66)
    print(f"  {'dense uniform @ 5 cm (grid_map)':<38}"
          f"{dense.n_x * dense.n_y:>14,}{dense.memory_bytes():>14,}")
    # cells_sensor_frame returns merged tiers keyed by level, not lattice
    # indices, so count by an accumulator every cell carries
    n_occ, n_fov = len(wm.c["ix"]), len(fov["n"])
    print(f"  {'occupied only, uniform @ 5 cm':<38}"
          f"{n_occ:>14,}{n_occ * SPARSE_BYTES:>14,}")
    print(f"  {'occupied only, foveated 5-40 cm':<38}"
          f"{n_fov:>14,}{n_fov * SPARSE_BYTES:>14,}")

    sparsity = (dense.n_x * dense.n_y) / max(n_occ, 1)
    foveation = n_occ / max(n_fov, 1)
    print(f"\n  sparsity gain (any hash map gives this) : {sparsity:8.1f}x")
    print(f"  foveation gain (this is the contribution): {foveation:8.2f}x")
    print(f"  combined against dense grid_map          : "
          f"{dense.memory_bytes() / max(n_fov * SPARSE_BYTES, 1):8.1f}x")

    lv, ct = np.unique(fov["lvl"], return_counts=True)
    print(f"\n  foveation tiers actually used:")
    for l, c in zip(lv, ct):
        print(f"    level {l}  {g.res0 * (1 << l) * 100:5.0f} cm  "
              f"{c:>8,} cells  {100 * c / n_fov:5.1f}%")

    print(f"\ngrid_map message: {a.msg_res:.2f} m over {a.extent:.0f} m square"
          f" -> {gm.n_x} x {gm.n_y} cells, {len(gm.layer_names)} layers")
    print(f"  occupancy {100 * gm.occupancy():.1f}%  "
          f"({gm.memory_bytes() / 1e6:.1f} MB dense, as grid_map would hold it)")

    if a.out:
        msg = gm.to_msg()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(msg))
        print(f"\nwrote {a.out}  ({a.out.stat().st_size / 1e6:.2f} MB)")
        print("  publish with rclpy: grid_map_msgs.msg.GridMap(**msg)")


if __name__ == "__main__":
    main()
