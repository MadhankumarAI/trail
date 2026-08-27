"""
KITTI 3D object detection: calibration, labels, and the camera <-> velodyne
transforms.

This module is the one place where frame conventions live. Getting them wrong
produces no error at all -- the loss still goes down and the results are quietly
nonsense -- so every transform here has an inverse and `selftest()` checks the
round trip.

Conventions
-----------
velodyne   x forward, y left,  z up      (origin at the laser)
camera     x right,   y down,  z forward (origin at cam 0)

A KITTI label line is:
    type trunc occ alpha  x1 y1 x2 y2  h w l  x y z  ry

where (x, y, z) is the **bottom centre** of the box in *camera* coordinates and
`ry` is rotation about the camera's y axis (which points down).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Classes we actually train on. Everything else in the file is either merged
# into these or excluded outright.
CLASSES = ["Background", "Car", "Pedestrian", "Cyclist"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# KITTI's own type strings mapped onto ours. Van and Truck are *not* merged into
# Car: the official benchmark does not, and merging inflates Car AP.
TYPE_MAP = {
    "Car": "Car",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
}
# Present in the labels but never a training target. 'DontCare' marks regions
# the annotators refused to label -- treating them as background poisons the
# negative set, so proposals overlapping them are dropped, not labelled 0.
IGNORE_TYPES = {"Van", "Truck", "Person_sitting", "Tram", "Misc", "DontCare"}


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@dataclass
class Calib:
    P2: np.ndarray          # (3, 4) left colour camera projection
    R0: np.ndarray          # (3, 3) rectifying rotation
    V2C: np.ndarray         # (3, 4) velodyne -> unrectified camera

    @classmethod
    def from_file(cls, path: Path | str) -> "Calib":
        vals = {}
        for line in Path(path).read_text().strip().splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            vals[k.strip()] = np.array([float(t) for t in v.split()])
        return cls(
            P2=vals["P2"].reshape(3, 4),
            R0=vals["R0_rect"].reshape(3, 3),
            V2C=vals["Tr_velo_to_cam"].reshape(3, 4),
        )

    # -- frame changes ------------------------------------------------------ #
    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) velodyne -> (N, 3) rectified camera."""
        h = np.hstack([pts, np.ones((len(pts), 1), pts.dtype)])
        return (self.R0 @ (self.V2C @ h.T)).T

    def cam_to_velo(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) rectified camera -> (N, 3) velodyne."""
        unrect = (np.linalg.inv(self.R0) @ pts.T).T
        R, t = self.V2C[:, :3], self.V2C[:, 3]
        # V2C is [R|t]; its inverse is R^T (x - t)
        return (R.T @ (unrect - t).T).T

    def project_to_image(self, pts_velo: np.ndarray) -> np.ndarray:
        """(N, 3) velodyne -> (N, 2) pixel coordinates in the left colour image."""
        cam = self.velo_to_cam(pts_velo)
        h = np.hstack([cam, np.ones((len(cam), 1), cam.dtype)])
        proj = (self.P2 @ h.T).T
        return proj[:, :2] / proj[:, 2:3]

    def fov_mask(self, pts_velo: np.ndarray,
                 img_w: int = 1242, img_h: int = 375) -> np.ndarray:
        """True for points the left colour camera can actually see.

        KITTI only annotates within this frustum. Points outside it are
        *unlabelled*, not background, so they must never become negatives.
        """
        cam = self.velo_to_cam(pts_velo)
        in_front = cam[:, 2] > 0.5
        uv = self.project_to_image(pts_velo)
        return (in_front
                & (uv[:, 0] >= 0) & (uv[:, 0] < img_w)
                & (uv[:, 1] >= 0) & (uv[:, 1] < img_h))


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
@dataclass
class Object3D:
    type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox2d: np.ndarray      # (4,) x1 y1 x2 y2
    h: float
    w: float
    l: float
    loc_cam: np.ndarray     # (3,) bottom centre, camera frame
    ry: float               # rotation about camera y

    @property
    def is_target(self) -> bool:
        return self.type in TYPE_MAP

    @property
    def class_id(self) -> int:
        return CLASS_TO_ID[TYPE_MAP[self.type]]

    def center_velo(self, calib: Calib) -> np.ndarray:
        """Geometric centre of the box in velodyne coordinates.

        The label stores the *bottom* centre and camera y points **down**, so
        the centre is h/2 in the negative y direction.
        """
        c = self.loc_cam.copy()
        c[1] -= self.h / 2.0
        return calib.cam_to_velo(c[None, :])[0]

    def yaw_velo(self, calib: Calib | None = None) -> float:
        """Heading about the velodyne z axis.

        Camera y (the ry axis) points down and maps to -z in velodyne, and the
        two frames differ by a further 90 degrees about that axis.
        """
        return -self.ry - np.pi / 2.0

    def dims_velo(self) -> np.ndarray:
        """(3,) extent along the box's own (length, width, height) axes."""
        return np.array([self.l, self.w, self.h], dtype=np.float64)

    def corners_velo(self, calib: Calib) -> np.ndarray:
        """(8, 3) box corners in velodyne coordinates."""
        l, w, h = self.l, self.w, self.h
        x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2.0
        y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2.0
        z = np.array([-h, -h, -h, -h, h, h, h, h]) / 2.0
        c, s = np.cos(self.yaw_velo()), np.sin(self.yaw_velo())
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return (R @ np.vstack([x, y, z])).T + self.center_velo(calib)


def read_labels(path: Path | str) -> List[Object3D]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().strip().splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        out.append(Object3D(
            type=f[0],
            truncation=float(f[1]),
            occlusion=int(float(f[2])),
            alpha=float(f[3]),
            bbox2d=np.array([float(v) for v in f[4:8]]),
            h=float(f[8]), w=float(f[9]), l=float(f[10]),
            loc_cam=np.array([float(v) for v in f[11:14]]),
            ry=float(f[14]),
        ))
    return out


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def points_in_box(pts: np.ndarray, center: np.ndarray, dims: np.ndarray,
                  yaw: float, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of which (N, 3) velodyne points fall inside an oriented box.

    Rotates the points into the box frame rather than the box into the world --
    same result, one rotation instead of eight.
    """
    d = pts[:, :3] - center
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx = d[:, 0] * c - d[:, 1] * s
    ly = d[:, 0] * s + d[:, 1] * c
    lz = d[:, 2]
    hl, hw, hh = dims / 2.0 + margin
    return (np.abs(lx) <= hl) & (np.abs(ly) <= hw) & (np.abs(lz) <= hh)


def read_velodyne(path: Path | str) -> np.ndarray:
    """(N, 4) float32 -- x, y, z, intensity."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def selftest(root: Path | str, frame: str = "000008") -> None:
    """Verify the transforms against real data rather than trusting them."""
    root = Path(root)
    calib = Calib.from_file(root / "training" / "calib" / f"{frame}.txt")
    objs = read_labels(root / "training" / "label_2" / f"{frame}.txt")
    velo_path = root / "training" / "velodyne" / f"{frame}.bin"

    print(f"frame {frame}: {len(objs)} labelled objects")

    # 1. round trip velo -> cam -> velo
    #
    # Tolerance is 0.1 mm, not machine epsilon. KITTI stores R0_rect and
    # Tr_velo_to_cam to 7 significant figures, so they are only orthonormal to
    # ~1e-7 (det(R0) = 0.99999994). That floor is a property of the dataset and
    # cannot be improved by better code. 0.1 mm is still ~100x tighter than
    # LiDAR range noise, so anything below it is irrelevant; anything above it
    # means a genuine bug.
    rng = np.random.default_rng(0)
    p = rng.normal(scale=10.0, size=(500, 3))
    back = calib.cam_to_velo(calib.velo_to_cam(p))
    err = np.abs(back - p).max()
    ortho = np.abs(calib.R0 @ calib.R0.T - np.eye(3)).max()
    print(f"  R0 orthonormality dev       {ortho:.2e}  (dataset precision floor)")
    print(f"  velo->cam->velo max error   {err:.2e} m   "
          f"{'OK' if err < 1e-4 else 'FAIL'}")
    assert err < 1e-4, f"frame round trip is broken: {err:.3e} m"

    if not velo_path.exists():
        print("  (velodyne not downloaded yet - skipping containment check)")
        return

    pts = read_velodyne(velo_path)
    fov = calib.fov_mask(pts[:, :3])
    print(f"  points {len(pts):,}   in camera FOV {fov.sum():,} "
          f"({100 * fov.mean():.1f}%)")

    # 2. do the transformed boxes actually contain points?
    print(f"  {'object':<12}{'range':>8}{'pts in box':>12}{'yaw':>9}")
    print("  " + "-" * 43)
    for o in objs:
        if not o.is_target:
            continue
        ctr = o.center_velo(calib)
        m = points_in_box(pts[:, :3], ctr, o.dims_velo(), o.yaw_velo())
        rng_m = float(np.linalg.norm(ctr[:2]))
        print(f"  {o.type:<12}{rng_m:>8.1f}{int(m.sum()):>12}"
              f"{np.degrees(o.yaw_velo()):>9.1f}")
        if rng_m < 40 and o.truncation < 0.5 and m.sum() == 0:
            raise AssertionError(
                f"{o.type} at {rng_m:.1f} m contains zero points - "
                "the box transform is wrong")
    print("  containment OK")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parents[2] / "data" / "kitti"
    selftest(root)
