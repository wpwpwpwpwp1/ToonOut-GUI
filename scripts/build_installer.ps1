param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $false)]
    [string]$IsccPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallerScript = Join-Path $ProjectRoot "installer\ToonOut.iss"
$AppExecutable = Join-Path $ProjectRoot "build\app\dist\ToonOut\ToonOut.exe"

if (-not (Test-Path -LiteralPath $AppExecutable -PathType Leaf)) {
    throw "Build the app before the installer: $AppExecutable"
}
if (-not $IsccPath) {
    $Command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($Command) {
        $IsccPath = $Command.Source
    } else {
        $IsccPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    }
}
if (-not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
    throw "Inno Setup compiler was not found: $IsccPath"
}

& $IsccPath "/DAppVersion=$Version" "/DProjectRoot=$ProjectRoot" $InstallerScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE."
}

$Output = Join-Path $ProjectRoot "dist\ToonOut-Setup-$Version.exe"
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
    throw "Installer was not created: $Output"
}
Write-Output "Installer created: $Output"
