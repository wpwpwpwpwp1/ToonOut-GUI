param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$CertificateBase64,
    [Parameter(Mandatory = $true)]
    [string]$CertificatePassword
)

$ErrorActionPreference = "Stop"
$Target = (Resolve-Path -LiteralPath $Path).Path
$SignTool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" `
    -Filter signtool.exe -Recurse -ErrorAction Stop |
    Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1
if (-not $SignTool) {
    throw "Windows SDK signtool.exe was not found."
}

$CertificatePath = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("toonout-codesign-{0}.pfx" -f [guid]::NewGuid())
try {
    [IO.File]::WriteAllBytes(
        $CertificatePath,
        [Convert]::FromBase64String($CertificateBase64)
    )
    & $SignTool.FullName sign /fd SHA256 /td SHA256 `
        /tr "http://timestamp.digicert.com" `
        /f $CertificatePath /p $CertificatePassword $Target
    if ($LASTEXITCODE -ne 0) {
        throw "Authenticode signing failed for $Target."
    }
    & $SignTool.FullName verify /pa /v $Target
    if ($LASTEXITCODE -ne 0) {
        throw "Authenticode verification failed for $Target."
    }
} finally {
    Remove-Item -LiteralPath $CertificatePath -Force -ErrorAction SilentlyContinue
}
