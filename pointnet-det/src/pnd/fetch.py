"""
Resumable chunked downloader for the KITTI 3D object detection dataset.

Why this exists instead of `curl -O`:

The KITTI S3 bucket throttles long-lived connections hard. A fresh connection
pulls at ~9 MB/s; the same connection after a couple of minutes decays to
~40 kB/s, which turns a 27 GB download into a week. Opening a new connection
per chunk keeps throughput at the route ceiling.

Parallelism past ~2 workers buys nothing (measured: 1 connection 9 MB/s,
6 connections 10 MB/s aggregate) because the bottleneck is bandwidth, not
concurrency. Workers default to 4 purely so one slow chunk cannot stall the
whole transfer.

Progress is journalled per chunk, so an interrupted download resumes instead of
starting over.

    python -m pnd.fetch                  # everything
    python -m pnd.fetch --only labels    # just the small files
    python -m pnd.fetch --workers 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

FILES = {
    "labels": (f"{BASE}/data_object_label_2.zip", "training/label_2"),
    "calib": (f"{BASE}/data_object_calib.zip", "training/calib"),
    "velodyne": (f"{BASE}/data_object_velodyne.zip", "training/velodyne"),
}

CHUNK = 64 << 20      # 64 MB per request - big enough to amortise setup,
                      # small enough that a stalled chunk is cheap to retry
RETRIES = 5


def _size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def _get_range(url: str, lo: int, hi: int, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _human(n: float) -> str:
    for u in ("B", "kB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def download(url: str, dest: Path, workers: int = 4) -> Path:
    """Download `url` to `dest`, resuming if a partial transfer exists."""
    total = _size(url)
    part = dest.with_suffix(dest.suffix + ".part")
    journal = dest.with_suffix(dest.suffix + ".progress")

    if dest.exists() and dest.stat().st_size == total:
        print(f"  {dest.name:<28} already complete ({_human(total)})")
        return dest

    chunks = [(i, i * CHUNK, min((i + 1) * CHUNK - 1, total - 1))
              for i in range((total + CHUNK - 1) // CHUNK)]

    done: set[int] = set()
    if journal.exists() and part.exists() and part.stat().st_size == total:
        try:
            done = set(json.loads(journal.read_text())["done"])
            print(f"  {dest.name:<28} resuming, {len(done)}/{len(chunks)} chunks present")
        except Exception:
            done = set()

    if not part.exists() or part.stat().st_size != total:
        with open(part, "wb") as f:      # preallocate so threads can seek
            f.truncate(total)
        done = set()

    todo = [c for c in chunks if c[0] not in done]
    if not todo:
        part.replace(dest)
        journal.unlink(missing_ok=True)
        return dest

    print(f"  {dest.name:<28} {_human(total)}  "
          f"{len(todo)} chunks of {_human(CHUNK)}  {workers} workers")

    start = time.time()
    got = len(done) * CHUNK
    lock_path = part

    def work(c):
        idx, lo, hi = c
        for attempt in range(RETRIES):
            try:
                buf = _get_range(url, lo, hi)
                if len(buf) != hi - lo + 1:
                    raise IOError(f"short read {len(buf)} != {hi - lo + 1}")
                with open(lock_path, "r+b") as f:
                    f.seek(lo)
                    f.write(buf)
                return idx, len(buf)
            except Exception as e:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            idx, nbytes = fut.result()
            done.add(idx)
            got += nbytes
            journal.write_text(json.dumps({"done": sorted(done)}))
            el = time.time() - start
            rate = (got - len(chunks) * 0) / max(el, 0.1)
            frac = got / total
            eta = (total - got) / max(rate, 1)
            sys.stdout.write(
                f"\r    {100*frac:5.1f}%  {_human(got)}/{_human(total)}  "
                f"{_human(rate)}/s  eta {eta/60:.1f} min   ")
            sys.stdout.flush()
    print()

    if part.stat().st_size != total:
        raise IOError(f"size mismatch: {part.stat().st_size} != {total}")
    part.replace(dest)
    journal.unlink(missing_ok=True)
    return dest


def extract(zip_path: Path, out_dir: Path, expect: str) -> None:
    marker = out_dir / expect
    if marker.exists() and any(marker.iterdir()):
        print(f"  {zip_path.name:<28} already extracted -> {expect}")
        return
    print(f"  {zip_path.name:<28} extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    n = sum(1 for _ in marker.iterdir()) if marker.exists() else 0
    print(f"    -> {expect}  ({n:,} files)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", choices=list(FILES),
                    help="fetch only these")
    ap.add_argument("--keep-zips", action="store_true",
                    help="do not delete archives after extraction")
    args = ap.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    kitti = root / "kitti"

    free = shutil.disk_usage(root).free
    want = 60 << 30
    print(f"disk free {_human(free)}  (need ~{_human(want)} for download + extract)")
    if free < want:
        print("  WARNING: this is tight. The zip is deleted after extraction "
              "unless --keep-zips.", file=sys.stderr)

    targets = args.only or list(FILES)
    print(f"\nfetching: {', '.join(targets)}\n")

    for name in targets:
        url, expect = FILES[name]
        zp = root / f"{name}.zip"
        download(url, zp, workers=args.workers)
        extract(zp, kitti, expect)
        if not args.keep_zips and name == "velodyne":
            zp.unlink(missing_ok=True)
            print(f"    removed {zp.name} to reclaim space")

    print("\nverifying ...")
    for sub, label in [("training/velodyne", "scans"),
                       ("training/label_2", "label files"),
                       ("training/calib", "calib files")]:
        d = kitti / sub
        n = sum(1 for _ in d.iterdir()) if d.exists() else 0
        print(f"  {sub:<24} {n:>7,} {label}")

    print("\ndone.")


if __name__ == "__main__":
    main()
