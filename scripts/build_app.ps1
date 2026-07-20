param(
    [Parameter(Mandatory = $false)]
    [string]$Version
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $ProjectRoot "build\app"
$DistRoot = Join-Path $BuildRoot "dist"
$WorkRoot = Join-Path $BuildRoot "work"
$SpecRoot = Join-Path $BuildRoot "spec"
$IconPath = Join-Path $ProjectRoot "assets\toonout.ico"
$MascotPath = Join-Path $ProjectRoot "assets\mascot"

if (-not (Test-Path -LiteralPath $IconPath -PathType Leaf)) {
    throw "App icon was not found: $IconPath"
}
if (-not (Test-Path -LiteralPath $MascotPath -PathType Container)) {
    throw "Mascot assets were not found: $MascotPath"
}

$DeclaredVersion = python -c "from app_version import APP_VERSION; print(APP_VERSION)"
if (-not $Version) {
    $Version = $DeclaredVersion
}
if ($Version -ne $DeclaredVersion) {
    throw "Requested version $Version does not match APP_VERSION $DeclaredVersion."
}

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
    --windowed `
    --name ToonOut `
    --icon $IconPath `
    --add-data "$IconPath;assets" `
    --add-data "$MascotPath;assets\mascot" `
    --distpath $DistRoot `
    --workpath $WorkRoot `
    --specpath $SpecRoot `
    --collect-all transformers `
    --collect-all timm `
    --collect-all kornia `
    --collect-all einops `
    --collect-all cryptography `
    (Join-Path $ProjectRoot "main.py")

$Executable = Join-Path $DistRoot "ToonOut\ToonOut.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "ToonOut.exe was not created: $Executable"
}
Write-Output "App created: $Executable"
