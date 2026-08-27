"""
Phase 2 - Anatomy of a LiDAR scan.

Takes ONE raw KITTI Velodyne file and pulls it completely apart, so you can see
exactly what a "point cloud" is as a data object. Nothing here is simulated.

Run:
    .venv\\Scripts\\python.exe scripts\\01_inspect_scan.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCANS = sorted((ROOT / "data" / "raw").rglob("velodyne_points/data/*.bin"))

SENSOR_HEIGHT = 1.73  # metres above ground, KITTI HDL-64E mounting


def rule(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    if not SCANS:
        sys.exit("No .bin scans found. Run scripts/00_fetch_data.py first.")

    path = SCANS[0]

    # ------------------------------------------------------------------ #
    rule("1. THE FILE ON DISK")
    # ------------------------------------------------------------------ #
    nbytes = path.stat().st_size
    print(f"path            {path.relative_to(ROOT)}")
    print(f"size            {nbytes:,} bytes  ({nbytes / 1e6:.2f} MB)")
    print()
    print("There is NO header. No magic bytes, no field names, no point count.")
    print("It is a flat wall of little-endian float32. You are expected to")
    print("already know the layout. That is the whole file format.")
    print()
    print(f"  {nbytes:,} bytes / 4 bytes per float32  = {nbytes // 4:,} floats")
    print(f"  {nbytes // 4:,} floats / 4 floats per point = {nbytes // 16:,} points")

    # ------------------------------------------------------------------ #
    rule("2. LOADING IT")
    # ------------------------------------------------------------------ #
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    print("pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)")
    print()
    print(f"shape           {pts.shape}      <- (N points, 4 channels)")
    print(f"dtype           {pts.dtype}")
    print(f"in RAM          {pts.nbytes / 1e6:.2f} MB")
    print()
    print("The 4 channels are x, y, z, intensity. In THIS order, in THIS frame:")
    print("  x  forward   (out of the windscreen)")
    print("  y  left")
    print("  z  up")
    print("  origin = the laser diode, 1.73 m above the road")
    print()
    print("Nothing in the file tells you that. It is convention. Get it wrong and")
    print("your scene is silently rotated 90 degrees and nothing will error.")

    # ------------------------------------------------------------------ #
    rule("3. FIVE ACTUAL POINTS")
    # ------------------------------------------------------------------ #
    print(f"{'idx':>8}  {'x (m)':>9} {'y (m)':>9} {'z (m)':>9} {'intens':>8}")
    print("-" * 78)
    for i in [0, 1, 2, len(pts) // 2, len(pts) - 1]:
        x, y, z, r = pts[i]
        print(f"{i:>8}  {x:>9.3f} {y:>9.3f} {z:>9.3f} {r:>8.3f}")
    print()
    print("That is a point. Four numbers. No colour, no normal, no object id,")
    print("no notion of which surface it belongs to, no link to its neighbours.")

    # ------------------------------------------------------------------ #
    rule("4. WHAT EACH CHANNEL ACTUALLY CONTAINS")
    # ------------------------------------------------------------------ #
    names = ["x", "y", "z", "intensity"]
    print(f"{'ch':<11}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}")
    print("-" * 78)
    for i, n in enumerate(names):
        c = pts[:, i]
        print(f"{n:<11}{c.min():>10.3f}{c.max():>10.3f}{c.mean():>10.3f}{c.std():>10.3f}")

    ground_z = -SENSOR_HEIGHT
    print()
    print(f"z bottoms out near {pts[:, 2].min():.2f} m. The road is at z = {ground_z:.2f} m,")
    print("because the origin is the sensor and the sensor is on the roof.")
    print("z is NOT height above ground until you add the mounting height.")
    print()
    print("intensity is in [0, 1] here: how much light came back. It is a")
    print("mixture of surface reflectivity, angle of incidence, and range")
    print("falloff -- all three baked into one number you cannot unmix.")

    # ------------------------------------------------------------------ #
    rule("5. THE MEASUREMENT THE SENSOR ACTUALLY MADE")
    # ------------------------------------------------------------------ #
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.sqrt(x**2 + y**2 + z**2)
    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arcsin(np.clip(z / np.maximum(rng, 1e-9), -1, 1)))

    print("x, y, z is NOT what the sensor measured. The sensor measured:")
    print("    range      how long the pulse took to come back")
    print("    azimuth    where the head was pointing when it fired")
    print("    elevation  which of the 64 stacked lasers fired")
    print()
    print("Cartesian is a derived convenience. The NATIVE format is spherical.")
    print("Remember this - it is the whole reason polar/cylindrical networks work.")
    print()
    print(f"{'':<12}{'min':>10}{'max':>10}{'mean':>10}")
    print("-" * 78)
    print(f"{'range (m)':<12}{rng.min():>10.2f}{rng.max():>10.2f}{rng.mean():>10.2f}")
    print(f"{'azimuth (d)':<12}{azim.min():>10.2f}{azim.max():>10.2f}{azim.mean():>10.2f}")
    print(f"{'elev (deg)':<12}{elev.min():>10.2f}{elev.max():>10.2f}{elev.mean():>10.2f}")

    # ------------------------------------------------------------------ #
    rule("6. THE HIDDEN STRUCTURE - FINDING THE 64 LASERS")
    # ------------------------------------------------------------------ #
    print("KITTI strips the laser id. But it is recoverable, because elevation")
    print("angle is quantised: each of the 64 lasers sits at a fixed angle.")
    print()
    hist, edges = np.histogram(elev, bins=400)
    peaks = int(np.sum(hist > len(pts) / 4000))
    print(f"elevation histogram over 400 bins -> {peaks} bins carry real mass")
    print("The cloud is not a formless blob. It is 64 nested cones of points.")
    print()
    print("Vertical spacing between adjacent lasers is ~0.4 deg. What that means")
    print("in metres depends entirely on how far away you are:")
    print()
    print(f"{'range':>9}   {'gap between adjacent rings':>28}")
    print("-" * 78)
    for r in [2, 5, 10, 20, 40, 80]:
        gap = r * np.tan(np.radians(0.4))
        print(f"{r:>7} m   {gap:>25.3f} m")
    print()
    print("A pedestrian at 5 m gets hit by ~50 rings. At 60 m, ~4 rings.")
    print("THAT is why far-away objects are unrecognisable, and it is the")
    print("physical argument for a grid that coarsens with distance.")

    # ------------------------------------------------------------------ #
    rule("7. DENSITY vs RANGE - MEASURED, NOT ASSUMED")
    # ------------------------------------------------------------------ #
    print(f"{'shell':>14}{'points':>12}{'volume m3':>13}{'pts/m3':>12}")
    print("-" * 78)
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]:
        m = (rng >= lo) & (rng < hi)
        vol = (2 / 3) * np.pi * (hi**3 - lo**3)  # hemisphere shell
        n = int(m.sum())
        print(f"{f'{lo}-{hi} m':>14}{n:>12,}{vol:>13,.0f}{n / vol:>12.3f}")
    print()
    print("Density falls off roughly as 1/r^2. A uniform 5 cm voxel grid spends")
    print("the same memory per cubic metre everywhere - so at 60 m it is storing")
    print("almost entirely empty cells. That is the waste your PS wants removed.")

    # ------------------------------------------------------------------ #
    rule("8. WHAT IT WOULD COST TO VOXELISE THIS NAIVELY")
    # ------------------------------------------------------------------ #
    for res in [0.05, 0.10, 0.20]:
        nx = int(160 / res)
        ny = int(160 / res)
        nz = int(10 / res)
        dense = nx * ny * nz
        occupied = len(np.unique((pts[:, :3] / res).astype(np.int32), axis=0))
        print(f"{res*100:>5.0f} cm voxels   dense grid {dense:>16,} cells"
              f"   actually occupied {occupied:>10,}"
              f"   ({100*occupied/dense:.4f}% full)")
    print()
    print("This single number is why sparse convolution exists.")

    print()
    print("=" * 78)
    print(f"  {len(pts):,} points. 4 numbers each. Everything else is inference.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
