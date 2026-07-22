param(
    [string[]]$GfxTargets = @(
        "gfx900", "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
        "gfx1010", "gfx1011", "gfx1012",
        "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
        "gfx1100", "gfx1101", "gfx1102", "gfx1103",
        "gfx1150", "gfx1151", "gfx1152", "gfx1153",
        "gfx1200", "gfx1201", "gfx1250"
    ),
    [string]$IndexUrl = "https://rocm.nightlies.amd.com/whl-multi-arch/",
    [switch]$KeepSourceArchive,
    [switch]$KeepBuildTree
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $ProjectRoot "build\amd-gpu-pack"
$DistRoot = Join-Path $BuildRoot "dist"
$WorkRoot = Join-Path $BuildRoot "work"
$SpecRoot = Join-Path $BuildRoot "spec"
$VirtualEnvironment = Join-Path $BuildRoot ".venv"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"
$WorkerOutput = Join-Path $DistRoot "ToonOutGpuWorker"
$PackOutput = Join-Path $ProjectRoot "dist\ToonOut-AMD-ROCm-GPU-Pack.zip"

if ($GfxTargets.Count -eq 0) {
    throw "At least one AMD gfx target is required."
}
foreach ($Target in $GfxTargets) {
    if ($Target -notmatch '^gfx[0-9a-z]+$') {
        throw "Invalid AMD gfx target: $Target"
    }
}

python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'"

if (Test-Path -LiteralPath $BuildRoot) {
    $ResolvedBuild = (Resolve-Path -LiteralPath $BuildRoot).Path
    $ExpectedParent = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "build")).Path
    if (-not $ResolvedBuild.StartsWith($ExpectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a build directory outside the project build folder."
    }
    Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force
}

python -m venv $VirtualEnvironment
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-inference.txt")
& $Python -m pip install "PyInstaller==6.21.0"

$DeviceExtras = ($GfxTargets | ForEach-Object { "device-$_" }) -join ","
$RocmPackages = @(
    "torch[$DeviceExtras]",
    "torchvision[$DeviceExtras]"
)
& $Python -m pip install --pre --index-url $IndexUrl @RocmPackages
& $Python -c "import torch; assert torch.version.hip; print(torch.__version__, torch.version.hip, torch.cuda.get_arch_list())"

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name ToonOutGpuWorker `
    --distpath $DistRoot `
    --workpath $WorkRoot `
    --specpath $SpecRoot `
    --collect-all torch `
    --collect-all torchvision `
    --collect-all transformers `
    --collect-all timm `
    --collect-all kornia `
    --collect-all einops `
    (Join-Path $ProjectRoot "gpu_worker.py")

$PackageArguments = @(
    (Join-Path $ProjectRoot "scripts\package_gpu_runtime.py"),
    "--source", $WorkerOutput,
    "--output", $PackOutput,
    "--vendor", "amd",
    "--backend", "rocm"
)
foreach ($Target in $GfxTargets) {
    $PackageArguments += @("--gfx-target", $Target)
}
& $Python @PackageArguments

$SplitArguments = @(
    (Join-Path $ProjectRoot "scripts\split_gpu_pack.py"),
    "--input", $PackOutput
)
if ($KeepSourceArchive) {
    $SplitArguments += "--keep-input"
}
& $Python @SplitArguments

& $Python -m pip freeze | Out-File `
    -FilePath (Join-Path $ProjectRoot "dist\ToonOut-AMD-ROCm-GPU-Pack.requirements.lock.txt") `
    -Encoding utf8

if (-not $KeepBuildTree -and (Test-Path -LiteralPath $BuildRoot)) {
    $ResolvedBuild = (Resolve-Path -LiteralPath $BuildRoot).Path
    $ExpectedParent = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "build")).Path
    if (-not $ResolvedBuild.StartsWith($ExpectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a build directory outside the project build folder."
    }
    Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force
}

Write-Output "AMD ROCm GPU pack parts and a dependency lock were created under dist."
