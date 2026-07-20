param(
    [switch]$KeepSourceArchive,
    [switch]$KeepBuildTree
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $ProjectRoot "build\gpu-pack"
$DistRoot = Join-Path $BuildRoot "dist"
$WorkRoot = Join-Path $BuildRoot "work"
$SpecRoot = Join-Path $BuildRoot "spec"
$WorkerOutput = Join-Path $DistRoot "ToonOutGpuWorker"
$PackOutput = Join-Path $ProjectRoot "dist\ToonOut-NVIDIA-GPU-Pack.zip"

python -c "import torch; assert torch.__version__.startswith('2.7.1+cu128'); assert torch.version.cuda == '12.8'; print(torch.__version__, torch.version.cuda)"

if (Test-Path -LiteralPath $BuildRoot) {
    $ResolvedBuild = (Resolve-Path -LiteralPath $BuildRoot).Path
    $ExpectedParent = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "build")).Path
    if (-not $ResolvedBuild.StartsWith($ExpectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a build directory outside the project build folder."
    }
    Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name ToonOutGpuWorker `
    --distpath $DistRoot `
    --workpath $WorkRoot `
    --specpath $SpecRoot `
    --collect-all transformers `
    --collect-all timm `
    --collect-all kornia `
    --collect-all einops `
    (Join-Path $ProjectRoot "gpu_worker.py")

python (Join-Path $ProjectRoot "scripts\package_gpu_runtime.py") `
    --source $WorkerOutput `
    --output $PackOutput

$SplitArguments = @(
    (Join-Path $ProjectRoot "scripts\split_gpu_pack.py"),
    "--input",
    $PackOutput
)
if ($KeepSourceArchive) {
    $SplitArguments += "--keep-input"
}
python @SplitArguments

if (-not $KeepBuildTree -and (Test-Path -LiteralPath $BuildRoot)) {
    $ResolvedBuild = (Resolve-Path -LiteralPath $BuildRoot).Path
    $ExpectedParent = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "build")).Path
    if (-not $ResolvedBuild.StartsWith($ExpectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a build directory outside the project build folder."
    }
    Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force
}

Write-Output "GPU pack Release parts created under dist."
