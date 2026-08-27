"""
Phase 2 follow-up: decode one point, find the ground empirically,
understand intensity, and prove the cloud is an unordered set.

Run:
    .venv\\Scripts\\python.exe scripts\\04_point_questions.py
"""
import sys
from pathlib import Path

import numpy as np

IDX = 119912
MOUNT = 1.73  # KITTI published Velodyne mounting height, in metres

# anchor to this file's location, not the shell's working directory
ROOT = Path(__file__).resolve().parent.parent
scans = sorted((ROOT / "data" / "raw").rglob("velodyne_points/data/*.bin"))
if not scans:
    sys.exit(
        f"No .bin scans found under {ROOT / 'data' / 'raw'}\n"
        "Download the data first:\n"
        '  curl.exe -L -o data\\kitti_raw_0001.zip '
        '"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'
        '2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"\n'
        "  Expand-Archive data\\kitti_raw_0001.zip -DestinationPath data\\raw -Force"
    )

p = scans[0]
d = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
x, y, z, inten = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
r = np.sqrt(x**2 + y**2 + z**2)


def rule(t):
    print("\n" + "=" * 74)
    print("  " + t)
    print("=" * 74)


# ------------------------------------------------------------------ #
rule("1. DECODING POINT #%d" % IDX)
# ------------------------------------------------------------------ #
px, py, pz, pi = d[IDX]
pr = np.sqrt(px**2 + py**2 + pz**2)
azi = np.degrees(np.arctan2(py, px))
elev = np.degrees(np.arcsin(pz / pr))
gdist = np.sqrt(px**2 + py**2)

print("stored in the file : x=%.3f  y=%.3f  z=%.3f  intensity=%.3f" % (px, py, pz, pi))
print()
print("everything else is DERIVED from those four, by arithmetic you can check:")
print("  range      sqrt(x^2 + y^2 + z^2)  = %.3f m" % pr)
print("  azimuth    atan2(y, x)            = %.2f deg" % azi)
print("  elevation  asin(z / range)        = %.2f deg" % elev)
print()
print("in plain words, that point is:")
print("  %.2f m away measured along the ground" % gdist)
print("  %.2f m in front of the car, %.2f m to the LEFT" % (px, py))
print("  %.2f m BELOW the laser" % abs(pz))
print("  fired by laser #62 of 64, aimed %.1f deg downward" % abs(elev))

# ------------------------------------------------------------------ #
rule("2. WHERE IS THE GROUND, ACTUALLY?  (measured, not assumed)")
# ------------------------------------------------------------------ #
near = r < 12
hist, edges = np.histogram(z[near], bins=400, range=(-3, 1))
modal_z = edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2

print("z is height relative to the LASER, not to the road.")
print("To find the road, look at where z piles up -- most near-field")
print("returns land on tarmac:")
print()
print("  modal z of returns within 12 m  = %.3f m" % modal_z)
print("  KITTI published mounting height = %.3f m" % MOUNT)
print("  the two agree to                = %.1f cm" % (abs(-modal_z - MOUNT) * 100))
print()
print("That is exactly how 'height AGL' on the page was computed:")
print("  height_AGL = z + %.2f = %.3f + %.2f = %.3f m" % (MOUNT, pz, MOUNT, pz + MOUNT))
print()
print("The 1.73 is NOT in the .bin file. It comes from KITTI's sensor setup")
print("documentation. The file on its own cannot tell you where the road is.")

flat = np.abs(z - modal_z) < 0.10
print()
print("points within 10 cm of that plane: %s of %s  (%.1f%%)"
      % (format(int(flat.sum()), ","), format(len(d), ","), 100 * flat.sum() / len(d)))

print()
print("But a CONSTANT ground height stops being true the moment the road")
print("is not perfectly flat and level. Fit a plane instead and compare:")
gm = flat & (r < 30)
A = np.c_[x[gm], y[gm], np.ones(int(gm.sum()))]
coef, *_ = np.linalg.lstsq(A, z[gm], rcond=None)
print("  fitted plane  z = %+.5f*x %+.5f*y %+.4f" % (coef[0], coef[1], coef[2]))
print("  road attitude    %+.3f deg pitch, %+.3f deg roll"
      % (np.degrees(np.arctan(coef[0])), np.degrees(np.arctan(coef[1]))))
print()
print("  %-12s %18s %18s" % ("band", "constant-height err", "fitted-plane err"))
print("  " + "-" * 50)
for lo, hi in [(0, 10), (10, 20), (20, 40)]:
    m = flat & (r >= lo) & (r < hi)
    if m.sum() < 50:
        continue
    pred = coef[0] * x[m] + coef[1] * y[m] + coef[2]
    print("  %-12s %15.1f cm %15.1f cm"
          % ("%d-%d m" % (lo, hi),
             np.abs(z[m] - modal_z).mean() * 100,
             np.abs(z[m] - pred).mean() * 100))

# ------------------------------------------------------------------ #
rule("3. WHAT IS INTENSITY?")
# ------------------------------------------------------------------ #
print("values in this scan run %.3f to %.3f" % (inten.min(), inten.max()))
print()
print("It is how much of the pulse came back. THREE things are baked in:")
print("  1. what the surface is made of   (paint vs matte tarmac vs glass)")
print("  2. the angle the beam struck it  (glancing hit = weak return)")
print("  3. how far away it was           (energy falls off with distance)")
print()
print("You cannot unmix them. Here is the SAME material -- tarmac -- at")
print("different ranges. The number moves anyway:")
print()
print("%10s %18s %12s" % ("range", "mean intensity", "points"))
print("-" * 42)
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]:
    m = flat & (r >= lo) & (r < hi)
    if m.sum() > 20:
        print("%10s %18.3f %12s"
              % ("%d-%d m" % (lo, hi), inten[m].mean(), format(int(m.sum()), ",")))

hi_g = flat & (inten > 0.55) & (r < 35)
print()
print("Still, the extremes are genuinely informative. Road paint is")
print("retroreflective, so it lights up. Ground returns with intensity")
print("> 0.55 inside 35 m: %s. Plotted top-down:" % format(int(hi_g.sum()), ","))
print()

XL, XH, YL, YH, C = 0.0, 32.0, -8.0, 8.0, 0.7
nx, ny = int((XH - XL) / C), int((YH - YL) / C)
grid = np.full((ny, nx), " ")

allg = flat & (r < 35)
ax_ = ((x[allg] - XL) / C).astype(int)
ay_ = ((y[allg] - YL) / C).astype(int)
ok = (ax_ >= 0) & (ax_ < nx) & (ay_ >= 0) & (ay_ < ny)
for a, b in zip(ay_[ok], ax_[ok]):
    grid[a, b] = "."

gx = ((x[hi_g] - XL) / C).astype(int)
gy = ((y[hi_g] - YL) / C).astype(int)
ok2 = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
for a, b in zip(gy[ok2], gx[ok2]):
    grid[a, b] = "#"

for i in range(ny - 1, -1, -1):
    print("  y=%+5.1f |%s" % (YL + i * C, "".join(grid[i])))
print("           " + "+" + "-" * nx)
print("            forward, 0 m to 32 m ->")
print()
print("  '#' = bright ground return (paint)     '.' = ordinary tarmac")

# ------------------------------------------------------------------ #
rule("4. WHY IT IS CALLED AN 'UNORDERED SET'")
# ------------------------------------------------------------------ #
print("An IMAGE stores position in the array INDEX. Pixel [5][6] sits next")
print("to [5][7] because of where it lives in memory. Shuffle the pixels and")
print("the picture is destroyed.")
print()
print("A POINT CLOUD stores position in the VALUES. The row number carries")
print("no meaning at all. Proof -- shuffle every row, then ask whether the")
print("scene changed:")
print()

rng_ = np.random.default_rng(0)
sh = d[rng_.permutation(len(d))]


def fingerprint(pts):
    v = np.unique((pts[:, :3] / 0.20).astype(np.int32), axis=0)
    return len(v), pts[:, :3].min(0), pts[:, :3].mean(0)


n1, mn1, c1 = fingerprint(d)
n2, mn2, c2 = fingerprint(sh)
print("  occupied 20 cm voxels   original %s     shuffled %s"
      % (format(n1, ","), format(n2, ",")))
print("  bounding-box min        %s  ->  %s"
      % (np.array2string(mn1, precision=3), np.array2string(mn2, precision=3)))
print("  centroid                %s  ->  %s"
      % (np.array2string(c1, precision=4), np.array2string(c2, precision=4)))
print()
print("  same scene?  %s" % (n1 == n2 and np.allclose(mn1, mn2) and np.allclose(c1, c2)))
print()
print("  orderings that all describe this identical scene:  %s factorial"
      % format(len(d), ","))
print("  (that is a number with roughly 570,000 digits)")

print()
print("And row-adjacency is NOT spatial adjacency. Compare the distance from")
print("each point to the NEXT ROW IN THE FILE against the distance to its")
print("true nearest neighbour in space:")
print()
step = np.linalg.norm(d[1:, :3] - d[:-1, :3], axis=1)
try:
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(d[:, :3].astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pc)
    samp = rng_.choice(len(d), 3000, replace=False)
    nn = []
    for i in samp:
        _, _, dd = tree.search_knn_vector_3d(pc.points[int(i)], 2)
        nn.append(np.sqrt(dd[1]))
    nn = np.array(nn)
    print("  median distance to the next row in the file : %.3f m" % np.median(step))
    print("  median distance to the true nearest neighbour: %.3f m" % np.median(nn))
    print("  file ordering is %.1fx worse than real adjacency"
          % (np.median(step) / np.median(nn)))
except Exception as e:
    print("  (open3d KD-tree unavailable: %s)" % e)
    print("  median distance to the next row in the file : %.3f m" % np.median(step))

print()
print("So a network cannot lean on array position. Whatever it computes must")
print("return the SAME answer for every ordering -- it must be PERMUTATION")
print("INVARIANT. Convolution is not. Max-pooling is.")
print("That is the single reason PointNet is built around max-pooling.")
print()
