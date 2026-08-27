# pointnet-det environment setup (Windows / PowerShell)
#
#   .\setup.ps1              detect GPU, install matching wheels
#   .\setup.ps1 -Cpu         force CPU-only torch
#   .\setup.ps1 -Cuda cu121  force a specific CUDA build
#
# Safe to re-run; it will not clobber an existing venv's packages unless
# -Force is passed.

param(
    [switch]$Cpu,
    [string]$Cuda = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "pointnet-det setup" -ForegroundColor Cyan
Write-Host ("-" * 60)

# ---------------------------------------------------------------- python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "python not found on PATH. Install Python 3.10-3.12 first." }

$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "python            $ver  ($($py.Source))"
if ([version]$ver -lt [version]"3.9") { throw "Python >= 3.9 required, found $ver" }

# ---------------------------------------------------------------- gpu probe
$variant = "cpu"
if ($Cuda) {
    $variant = $Cuda
    Write-Host "gpu               forced -> $variant"
}
elseif ($Cpu) {
    Write-Host "gpu               forced CPU"
}
else {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $name = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
        $drv = [double]((& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1) -split '\.')[0]
        Write-Host "gpu               $name  (driver $drv)"
        # driver >= 525 supports the cu121 runtime; older gets cu118
        if ($drv -ge 525) { $variant = "cu121" } else { $variant = "cu118" }
        Write-Host "torch build       $variant"
    }
    else {
        Write-Host "gpu               none detected -> CPU wheels" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- venv
if (-not (Test-Path ".venv")) {
    Write-Host "`ncreating .venv ..."
    & python -m venv .venv
}
else {
    Write-Host "`n.venv already exists"
}
$vpy = Join-Path $Root ".venv\Scripts\python.exe"

& $vpy -m pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------- torch
$installed = & $vpy -c "
try:
    import torch; print(torch.__version__)
except Exception:
    print('')
"
if ($installed -and -not $Force) {
    Write-Host "torch             already installed ($installed) - skipping"
}
else {
    Write-Host "`ninstalling torch ($variant) ..."
    if ($variant -eq "cpu") {
        & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
    else {
        & $vpy -m pip install torch --index-url "https://download.pytorch.org/whl/$variant"
    }
}

# ---------------------------------------------------------------- rest
Write-Host "`ninstalling the rest ..."
& $vpy -m pip install --quiet -r requirements.txt
& $vpy -m pip install --quiet -e .

# ---------------------------------------------------------------- verify
Write-Host "`n$('-' * 60)"
& $vpy -c @"
import torch, numpy, numba, sys
print(f'python   {sys.version.split()[0]}')
print(f'numpy    {numpy.__version__}')
print(f'numba    {numba.__version__}')
print(f'torch    {torch.__version__}')
if torch.cuda.is_available():
    i = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f'cuda     yes - {i.name}')
    print(f'         {i.total_memory/1e9:.1f} GB, compute {cap[0]}.{cap[1]}')
    print(f'         bf16 {\"yes\" if cap[0] >= 8 else \"no (Ampere+ only)\"}')
else:
    print('cuda     no - running on CPU')
"@

Write-Host "`nready." -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -m pnd.fetch"
