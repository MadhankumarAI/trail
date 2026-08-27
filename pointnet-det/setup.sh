#!/usr/bin/env bash
# pointnet-det environment setup (Linux / macOS)
#
#   ./setup.sh              detect GPU, install matching wheels
#   ./setup.sh --cpu        force CPU-only torch
#   ./setup.sh --cuda cu121 force a specific CUDA build
set -euo pipefail
cd "$(dirname "$0")"

FORCE_CPU=0; CUDA=""; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) FORCE_CPU=1; shift;;
    --cuda) CUDA="$2"; shift 2;;
    --force) FORCE=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

echo "pointnet-det setup"
printf '%.0s-' {1..60}; echo

command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
VER=$(python3 -c "import sys; print('%d.%d' % sys.version_info[:2])")
echo "python            $VER  ($(command -v python3))"

VARIANT="cpu"
if [[ -n "$CUDA" ]]; then
  VARIANT="$CUDA"; echo "gpu               forced -> $VARIANT"
elif [[ $FORCE_CPU -eq 1 ]]; then
  echo "gpu               forced CPU"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  echo "gpu               $NAME  (driver $DRV)"
  if [[ "$DRV" -ge 525 ]]; then VARIANT="cu121"; else VARIANT="cu118"; fi
  echo "torch build       $VARIANT"
else
  echo "gpu               none detected -> CPU wheels"
fi

[[ -d .venv ]] || { echo; echo "creating .venv ..."; python3 -m venv .venv; }
VPY=".venv/bin/python"
$VPY -m pip install --quiet --upgrade pip setuptools wheel

HAVE=$($VPY -c "
try:
    import torch; print(torch.__version__)
except Exception:
    print('')
")
if [[ -n "$HAVE" && $FORCE -eq 0 ]]; then
  echo "torch             already installed ($HAVE) - skipping"
else
  echo; echo "installing torch ($VARIANT) ..."
  if [[ "$VARIANT" == "cpu" ]]; then
    $VPY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  else
    $VPY -m pip install torch --index-url "https://download.pytorch.org/whl/$VARIANT"
  fi
fi

echo; echo "installing the rest ..."
$VPY -m pip install --quiet -r requirements.txt
$VPY -m pip install --quiet -e .

echo; printf '%.0s-' {1..60}; echo
$VPY - <<'PYEOF'
import sys, numpy, numba, torch
print(f"python   {sys.version.split()[0]}")
print(f"numpy    {numpy.__version__}")
print(f"numba    {numba.__version__}")
print(f"torch    {torch.__version__}")
if torch.cuda.is_available():
    i = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"cuda     yes - {i.name}")
    print(f"         {i.total_memory/1e9:.1f} GB, compute {cap[0]}.{cap[1]}")
    print(f"         bf16 {'yes' if cap[0] >= 8 else 'no (Ampere+ only)'}")
else:
    print("cuda     no - running on CPU")
PYEOF

echo; echo "ready."
echo "  .venv/bin/python -m pnd.fetch"
