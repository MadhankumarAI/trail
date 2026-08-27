# pointnet-det

LiDAR 3D object detection on KITTI via **ground removal → clustering → PointNet
per cluster**, with a deterministic canonicalisation stage replacing PointNet's
input T-Net.

The research question this repo exists to answer:

> Can a cheap **deterministic** canonicalisation replace the learned input
> T-Net without losing accuracy — and if so, which one?

Four variants are implemented behind one config flag and measured head to head:

| `canon` | What it does | Params | Cost |
|---|---|---|---|
| `none` | Centre + scale only | 0 | ~0 |
| `tnet3` | Learned 3×3 transform (original PointNet) | ~800 k | GPU |
| `pca3_skew` | Full 3D PCA, sign fixed by third moment | 0 | CPU/GPU, deterministic |
| `pca2_yaw` | 2D PCA on (x, y) — yaw only, gravity preserved | 0 | CPU/GPU, deterministic |

The 64×64 **feature** T-Net from the original paper is removed in all variants.
It is the expensive one (4,096 regressed outputs plus an orthogonality
regulariser) and buys ~2% on clean CAD data. PCA cannot replace it — PCA is a
geometric operation and the feature space has no geometry — so it is simply
deleted.

## Quick start

**Windows**

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe -m pnd.fetch          # downloads KITTI (~27.5 GB)
.\.venv\Scripts\python.exe -m pnd.preprocess     # cache proposals
.\.venv\Scripts\python.exe -m pnd.train --config configs/pca2_yaw.yaml
```

**Linux / the A100 box**

```bash
./setup.sh
.venv/bin/python -m pnd.fetch
.venv/bin/python -m pnd.preprocess
.venv/bin/python -m pnd.train --config configs/pca2_yaw.yaml
```

Run the full ablation:

```bash
.venv/bin/python -m pnd.ablation      # trains all four, writes results table
```

## Hardware adaptation

`setup` detects the GPU and pins the right wheels; `pnd.config` then adapts at
runtime:

| GPU | Compute | Precision | Batch | `torch.compile` |
|---|---|---|---|---|
| A100 80 GB | sm_80 | bf16 + TF32 | 512 | on |
| GTX 1650 4 GB | sm_75 | fp16 AMP | 64 | off |
| CPU | — | fp32 | 32 | off |

bf16 is Ampere-and-later only — a Turing card silently falls back and gets
slower, so the dtype is chosen from the detected capability rather than assumed.

## Layout

```
configs/            one YAML per ablation arm
src/pnd/
  kitti.py          calib parsing, camera↔velodyne box transforms
  ground.py         numba ground-plane removal
  cluster.py        numba voxel connected components
  proposals.py      clusters → labelled proposals
  canon.py          the four canonicalisation variants
  features.py       eigen-features, height AGL, range, intensity
  model.py          PointNet backbone + classification and box heads
  train.py          training loop
  evaluate.py       KITTI AP
  bench.py          end-to-end latency
data/kitti/         downloaded dataset
cache/              preprocessed proposals
runs/               checkpoints and metrics
```

## Known limits

**Recall is capped by the clustering, not the network.** Objects that merge into
one cluster, or fragment into several, are lost before PointNet sees them. This
is the fundamental ceiling of the cluster-then-classify design and it is
reported explicitly in `evaluate.py` as a separate number from the classifier's
accuracy, so the two failure modes never get confused.

**KITTI only annotates the front camera field of view.** Points outside roughly
±40° of forward are unlabelled — *not* background. Training on them as
background would poison the negative set, so they are excluded. `DontCare`
regions are excluded for the same reason.
