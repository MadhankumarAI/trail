# SIH 2026 — PS 26053
Adaptive Variable Resolution 2.5D Lidar Mapping

## Setup (already done — only needed on a fresh machine)

```powershell
cd c:\Users\jaip7\Downloads\madhan\sih26
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy open3d matplotlib pillow
```

## Data

KITTI raw, drive `2011_09_26_0001` — 108 Velodyne HDL-64E scans + camera + GPS/IMU.
Already downloaded to `data/raw/`. To re-fetch:

```powershell
curl.exe -L -o data\kitti_raw_0001.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"
Expand-Archive data\kitti_raw_0001.zip -DestinationPath data\raw -Force
```

## Commands

| What | Command |
|---|---|
| Tear one scan apart, print everything | `.\.venv\Scripts\python.exe scripts\01_inspect_scan.py` |
| Pack a scan for the browser | `.\.venv\Scripts\python.exe scripts\02_export_web.py` |
| Rebuild the Phase 2 page | `.\.venv\Scripts\python.exe scripts\03_build_page.py` |

`02` then `03` must run in that order — `02` writes `web/payload.json.txt`, `03` injects it.

## Pages — open these in a browser

| File | Phase |
|---|---|
| `phases\01-foundations.html` | 1 — 2D / 2.5D / 3D foundations |
| `phases\02-point-anatomy.html` | 2 — what a point cloud is |

```powershell
start .\phases\01-foundations.html
start .\phases\02-point-anatomy.html
```

Both are fully self-contained (scan data is embedded). No server needed.

**Do not open `web\template_02.html`** — it is the source template and still has
`__PAYLOAD__` placeholders in it. `scripts\03_build_page.py` turns it into
`phases\02-point-anatomy.html`, which is the one you actually view.

## Layout

```
.venv/      virtual environment
data/raw/   KITTI scans (.bin, 4 float32 per point: x y z intensity)
scripts/    numbered, run in order
web/        build intermediates — not for viewing
phases/     the pages you open
```
