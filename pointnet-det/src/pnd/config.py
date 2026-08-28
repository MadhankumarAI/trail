"""
Config plus hardware adaptation.

One YAML runs unchanged on a 4 GB Turing laptop and an 80 GB A100. The parts
that must differ -- batch size, autocast dtype, whether torch.compile is worth
it -- are derived from the detected device rather than written down, because a
config file that has to be edited per machine gets edited wrong.

The one non-obvious rule: bf16 requires compute capability >= 8.0 (Ampere).
A Turing card (7.5) will silently accept `torch.autocast(dtype=bfloat16)` and
then run it slowly in software, so the dtype is chosen from the capability.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from .canon import MODES

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- data -------------------------------------------------------- #
    data_root: Path = ROOT / "data" / "kitti"
    cache_dir: Path = ROOT / "cache"
    run_dir: Path = ROOT / "runs"
    max_frames: Optional[int] = None      # None = all 7481
    val_frac: float = 0.2

    # --- proposals --------------------------------------------------- #
    ground_thresh: float = 0.25
    cluster_voxel: float = 0.30
    min_cluster_pts: int = 20
    max_cluster_pts: int = 6000
    max_range: float = 50.0
    n_points: int = 256                   # points sampled per proposal
    fg_iou_pts: float = 0.5               # frac of cluster inside a box -> that class
    bg_max_pts: float = 0.05              # below this -> background

    # --- model ------------------------------------------------------- #
    canon: str = "pca2_yaw"
    num_classes: int = 4
    in_ch: int = 6                        # xyz + agl + range + intensity
    width: int = 1024
    dropout: float = 0.3

    # --- training ---------------------------------------------------- #
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    box_loss_w: float = 1.0
    corner_loss_w: float = 0.5   # regulariser weight, not a replacement
    seed: int = 0
    tag: Optional[str] = None   # run dir name; keeps seeds apart
    num_workers: int = 4

    # --- augmentation -------------------------------------------------#
    # 84% of clusters hold fewer than n_points real points (median 55), so the
    # network mostly sees duplicated points. These augmentations put genuine
    # variation back in. Point dropout is the important one: it simulates the
    # 5000x density falloff measured between the near field and 40-80 m, so the
    # model stops assuming an object always arrives with the same point budget.
    yaw_aug: bool = True                  # random heading
    aug_dropout: float = 0.4              # max fraction of points discarded
    aug_jitter: float = 0.02              # per-point gaussian noise, metres
    aug_scale: float = 0.08               # +/- fraction
    aug_flip: bool = True                 # mirror about the x axis

    # --- loss ---------------------------------------------------------#
    # "inv" gave weights [0.28, 2.46, 20, 20]; recall hit 0.97 and precision
    # collapsed to 0.65. "sqrt_inv" is the gentler standard alternative.
    weight_mode: str = "sqrt_inv"         # inv | sqrt_inv | none
    weight_clip: float = 8.0
    label_smoothing: float = 0.05
    ema_decay: float = 0.999              # 0 disables

    # --- filled in by detect() --------------------------------------- #
    device: str = "cpu"
    batch_size: int = 32
    amp_dtype: Optional[str] = None
    compile: bool = False
    gpu_name: str = ""

    def detect(self) -> "Config":
        import torch
        if not torch.cuda.is_available():
            self.device, self.batch_size = "cpu", 32
            self.amp_dtype, self.compile = None, False
            self.gpu_name = "cpu"
            return self

        self.device = "cuda"
        major, minor = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        self.gpu_name = props.name
        vram = props.total_memory / 1e9

        if major >= 8:                              # Ampere and later
            self.amp_dtype = "bfloat16"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # TorchDynamo, which backs torch.compile, has no Python 3.13
            # support: torch.compile() raises outright rather than degrading.
            # Capability alone is not enough to decide this.
            self.compile = sys.version_info < (3, 13)
        else:                                       # Turing and earlier
            self.amp_dtype = "float16"
            self.compile = False

        # the ensemble arm runs 4 copies per sample, so it needs a quarter batch
        div = 4 if self.canon == "pca4_ensemble" else 1
        if vram >= 40:
            self.batch_size = 512 // div
        elif vram >= 20:
            self.batch_size = 256 // div
        elif vram >= 10:
            self.batch_size = 128 // div
        else:
            self.batch_size = 64 // div
        return self

    @classmethod
    def load(cls, path: Optional[Path] = None, **over: Any) -> "Config":
        vals: dict = {}
        if path is not None:
            vals = yaml.safe_load(Path(path).read_text()) or {}
        vals.update({k: v for k, v in over.items() if v is not None})
        for k in ("data_root", "cache_dir", "run_dir"):
            if k in vals:
                vals[k] = Path(vals[k])
        cfg = cls(**vals)
        if cfg.canon not in MODES + ("pca4_ensemble",):
            raise ValueError(f"canon must be one of {MODES + ('pca4_ensemble',)}, "
                             f"got {cfg.canon!r}")
        return cfg.detect()

    def summary(self) -> str:
        return (f"canon={self.canon}  device={self.device} ({self.gpu_name})\n"
                f"batch={self.batch_size}  amp={self.amp_dtype}  "
                f"compile={self.compile}\n"
                f"epochs={self.epochs}  lr={self.lr}  points={self.n_points}")

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
