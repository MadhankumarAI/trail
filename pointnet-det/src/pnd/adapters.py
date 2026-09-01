"""
Input adapters. Every sensor source converts here and nowhere else.

THE CANONICAL CONTRACT
----------------------
Everything downstream -- ground.py, terrain.py, cluster.py, model.py -- assumes
exactly this and nothing more:

    points   (N, 4) float32   x forward, y LEFT, z up, intensity in [0, 1]
    origin   the sensor itself, so z is height relative to the LiDAR
    height   sensor height above the road, in metres, supplied separately

That is ROS REP-103 and it is what KITTI's Velodyne files already use. Any
source that differs is converted here, once.

WHY THIS FILE EXISTS: CARLA IS LEFT-HANDED
------------------------------------------
CARLA's coordinate system is x forward, **y RIGHT**, z up. ROS and KITTI are
x forward, **y LEFT**, z up. Feeding CARLA points in unconverted mirrors the
entire scene: nothing raises, no shape is wrong, the loss still falls. What you
get is a model quietly trained on a mirror world, with headings negated and
left-turn geometry learned as right-turn geometry.

`from_carla` negates y. If your CARLA capture script already converted to a
right-handed frame, pass `already_rhs=True` -- and check, do not assume, because
both mistakes look identical in a plot.

OTHER THINGS THAT DIFFER AND WILL BITE
--------------------------------------
1. DENSITY. The manifest shows CARLA at ~25,600 points per sweep against
   KITTI's ~121,000 -- 4.7x fewer. Clustering parameters tuned on KITTI
   (min_cluster_pts=20, cluster_voxel=0.30) are wrong at that density: the same
   object now returns a quarter of the points and falls under the minimum.
   `suggest_params` scales them.

2. SENSOR HEIGHT. Hard-coded 1.73 m everywhere because that is where KITTI
   mounts its Velodyne. CARLA's height is whatever the spawn transform says.
   Ground removal's sanity check rejects planes far from -sensor_h, so a wrong
   height silently disables it.

3. INTENSITY. CARLA's raw LiDAR gives intensity in [0, 1] from an attenuation
   model, not a real reflectance. CARLA's *semantic* LiDAR gives no intensity at
   all -- its fields are (x, y, z, cos_inc_angle, object_idx, object_tag). If
   you feed semantic-LiDAR points, channel 5 of the network input is not
   intensity and the model is being fed a different quantity than it trained on.

4. DOMAIN GAP. The checkpoint was trained on real KITTI. CARLA point clouds are
   noiseless and geometrically ideal. Expect degradation; measure it before
   trusting it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# KITTI mounts its Velodyne here. Everything else must say so explicitly.
KITTI_SENSOR_H = 1.73


class ScanError(ValueError):
    """Raised when an input violates the canonical contract."""


# --------------------------------------------------------------------------- #
def validate(pts: np.ndarray, sensor_h: float, name: str = "scan") -> dict:
    """Check a scan against the contract and report what is suspicious.

    Deliberately loud. Every failure mode this catches is one that otherwise
    produces no error at all -- just worse numbers that get blamed on the model.
    """
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ScanError(f"{name}: expected (N, >=3), got {pts.shape}")
    if len(pts) < 100:
        raise ScanError(f"{name}: only {len(pts)} points")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rep: dict = {"n": len(pts), "warnings": []}
    W = rep["warnings"].append

    # Ground should sit near -sensor_h. If it does not, either the height is
    # wrong or z is not measured from the sensor.
    hist, edges = np.histogram(z, bins=200, range=(-5, 5))
    modal_z = edges[int(np.argmax(hist))]
    rep["modal_z"] = float(modal_z)
    rep["implied_sensor_h"] = float(-modal_z)
    if abs(-modal_z - sensor_h) > 0.5:
        W(f"ground looks like it is at z={modal_z:.2f}, implying a sensor "
          f"height of {-modal_z:.2f} m, but {sensor_h:.2f} m was given. "
          f"Ground removal's sanity check will misfire.")

    if pts.shape[1] >= 4:
        i = pts[:, 3]
        rep["intensity"] = (float(i.min()), float(i.max()))
        if i.max() > 1.001:
            W(f"intensity max is {i.max():.2f}; the model was trained on [0, 1]")
        if i.max() - i.min() < 1e-6:
            W("intensity is constant -- channel 5 carries no information")
    else:
        W("no intensity channel")

    rng = np.sqrt(x ** 2 + y ** 2)
    rep["max_range"] = float(rng.max())
    rep["density"] = float(len(pts))
    if len(pts) < 60000:
        W(f"{len(pts):,} points per sweep. KITTI has ~121,000; clustering "
          f"parameters tuned there will under-segment. See suggest_params().")

    # A left-handed frame fed in unconverted is not detectable from one scan --
    # the cloud looks equally plausible mirrored. Say so rather than pretend.
    rep["handedness_checkable"] = False
    return rep


def report(rep: dict, name: str = "scan") -> None:
    print(f"{name}: {rep['n']:,} points, max range {rep['max_range']:.1f} m, "
          f"ground implies sensor height {rep['implied_sensor_h']:.2f} m")
    if rep.get("intensity"):
        print(f"  intensity {rep['intensity'][0]:.3f} .. {rep['intensity'][1]:.3f}")
    for w in rep["warnings"]:
        print(f"  WARNING: {w}")
    if not rep["warnings"]:
        print("  no warnings")


# --------------------------------------------------------------------------- #
def from_kitti(path: Path | str) -> np.ndarray:
    """KITTI .bin -- already canonical. (N, 4) float32."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def from_carla(path: Path | str,
               already_rhs: bool = False,
               intensity_key: Optional[str] = None) -> dict:
    """CARLA .npz -> canonical points, plus semantic tags when present.

    Returns {"points": (N, 4), "tags": (N,) or None, "obj_idx": (N,) or None}.

    The npz key names are not fixed by CARLA -- they are whatever the capture
    script chose -- so this looks for the usual candidates and fails loudly
    rather than silently picking the wrong array.
    """
    z = np.load(str(path), allow_pickle=True)
    keys = list(z.keys())

    def pick(cands, required=True):
        for c in cands:
            if c in z:
                return z[c]
        if required:
            raise ScanError(
                f"{path}: none of {cands} found. Keys present: {keys}")
        return None

    arr = pick(["points", "data", "xyz", "lidar", "arr_0"])
    arr = np.asarray(arr)
    if arr.dtype.names:                      # structured array
        cols = arr.dtype.names
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
        inten = arr["intensity"] if "intensity" in cols else None
        tags = arr["ObjTag"] if "ObjTag" in cols else (
            arr["object_tag"] if "object_tag" in cols else None)
        oidx = arr["ObjIdx"] if "ObjIdx" in cols else (
            arr["object_idx"] if "object_idx" in cols else None)
    else:
        arr = arr.reshape(len(arr), -1)
        xyz = arr[:, :3]
        # raw LiDAR is (x, y, z, intensity); semantic LiDAR is
        # (x, y, z, cos_inc_angle, object_idx, object_tag)
        if arr.shape[1] == 4:
            inten, tags, oidx = arr[:, 3], None, None
        elif arr.shape[1] >= 6:
            inten, oidx, tags = None, arr[:, 4], arr[:, 5]
        else:
            inten = tags = oidx = None
        tags = pick(["tags", "obj_tag", "semantic"], required=False) \
            if tags is None else tags
        inten = pick([intensity_key], required=False) \
            if intensity_key else inten

    xyz = np.asarray(xyz, np.float32)
    if not already_rhs:
        # CARLA is x forward, y RIGHT, z up. Canonical is y LEFT.
        xyz = xyz.copy()
        xyz[:, 1] *= -1.0

    if inten is None:
        # Semantic LiDAR has no intensity. Zero is honest -- it tells the model
        # nothing, which is better than feeding it cos_inc_angle and calling it
        # reflectance.
        inten = np.zeros(len(xyz), np.float32)
    inten = np.asarray(inten, np.float32).reshape(-1)
    if inten.max() > 1.001:
        inten = inten / max(inten.max(), 1e-6)

    return {
        "points": np.column_stack([xyz, inten]).astype(np.float32),
        "tags": None if tags is None else np.asarray(tags).reshape(-1).astype(np.int32),
        "obj_idx": None if oidx is None else np.asarray(oidx).reshape(-1).astype(np.int32),
        "keys": keys,
    }


def from_ros2(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 -> canonical points.

    ROS is already REP-103, so no axis flip. The catch is that PointCloud2 is a
    byte buffer with a self-describing field layout: field order, offsets and
    padding vary by publisher, so the fields must be read by name rather than
    assumed positional.
    """
    try:
        from sensor_msgs_py import point_cloud2 as pc2
    except ImportError as e:
        raise ScanError(
            "from_ros2 needs sensor_msgs_py (part of a ROS 2 install)") from e

    names = [f.name for f in msg.fields]
    want = ["x", "y", "z"] + (["intensity"] if "intensity" in names else [])
    arr = pc2.read_points_numpy(msg, field_names=want, skip_nans=True)
    if arr.shape[1] == 3:
        arr = np.column_stack([arr, np.zeros(len(arr), np.float32)])
    return arr.astype(np.float32)


# --------------------------------------------------------------------------- #
def suggest_params(n_points: int, cfg=None) -> dict:
    """Scale density-dependent parameters from KITTI's ~121k points.

    Clustering thresholds are counts, so they do not transfer across sensors.
    At CARLA's ~25,600 points the same object returns roughly a quarter as many
    returns, and a min_cluster_pts of 20 tuned on KITTI silently discards
    objects that are perfectly well observed.
    """
    ratio = n_points / 121000.0
    out = {
        "density_ratio": round(ratio, 3),
        "min_cluster_pts": max(6, int(round(20 * ratio))),
        "n_points": max(64, int(2 ** round(np.log2(max(256 * ratio, 64))))),
        # coarser voxel: fewer points must still connect into one component
        "cluster_voxel": round(float(np.clip(0.30 / max(ratio, 0.05) ** 0.5,
                                             0.30, 0.80)), 2),
    }
    return out
