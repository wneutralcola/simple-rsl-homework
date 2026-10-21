param(
    [string]$RuntimeDirectory = "",
    [string]$ArchivePath = "",
    [switch]$SkipPull
)
$ErrorActionPreference = 'Stop'
if (-not $RuntimeDirectory) {
    $RuntimeDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.runtime\ollama'))
}
New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
$version = '0.40.3'
$expectedHash = 'e5ecff02dfc7cae8ddf6d6b7a7ac5f727ffeedf4491f44ae75ff5b0820dbbbd5'
$executable = Join-Path $RuntimeDirectory 'ollama.exe'
if (-not (Test-Path -LiteralPath $executable)) {
    if (-not $ArchivePath) {
        $ArchivePath = Join-Path $RuntimeDirectory "ollama-windows-amd64-v$version.zip"
        $downloadStamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $url = "https://github.com/ollama/ollama/releases/download/v$version/ollama-windows-amd64.zip?download=$downloadStamp"
        Write-Host "Downloading official Ollama portable runtime ($version)..."
        & curl.exe -L --fail --retry 3 --retry-all-errors --retry-delay 2 --continue-at - --connect-timeout 20 --output $ArchivePath $url
        if ($LASTEXITCODE -ne 0) { throw 'Ollama download failed.' }
    }
    if ((Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
        throw 'Archive SHA256 does not match the official release. Do not extract it.'
    }
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $RuntimeDirectory -Force
}
& (Join-Path $PSScriptRoot 'start.ps1') -RuntimeDirectory $RuntimeDirectory
if (-not $SkipPull) {
    $env:OLLAMA_HOST = '127.0.0.1:11435'
    & $executable pull 'qwen3:4b-q4_K_M'
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed; rerun setup.ps1 to resume.' }
}
Write-Host 'Ready: http://127.0.0.1:11435 . Run: .\run.ps1 --host http://127.0.0.1:11435 demo --dataset dataset.json'
