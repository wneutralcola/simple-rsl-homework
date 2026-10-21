param([string]$RuntimeDirectory = "")
$ErrorActionPreference = 'Stop'
if (-not $RuntimeDirectory) {
    $RuntimeDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.runtime\ollama'))
}
$executable = Join-Path $RuntimeDirectory 'ollama.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw 'Run setup.ps1 first.' }
$pidFile = Join-Path $RuntimeDirectory 'server.pid'
$existing = $null
if (Test-Path -LiteralPath $pidFile) {
    $serverPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $existing = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
}
try {
    $service = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 2
} catch { $service = $null }
if ($service) {
    if ($existing -and $existing.Path -eq $executable) {
        Write-Host "Ollama already running (PID $($existing.Id), version $($service.version))."
        return
    }
    throw 'Port 11435 is already used by another Ollama service.'
}
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $RuntimeDirectory 'models'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NOHISTORY = '1'
if ($existing -and $existing.Path -eq $executable) {
    $process = $existing
} else {
    $process = Start-Process -FilePath $executable -ArgumentList 'serve' -WorkingDirectory $RuntimeDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeDirectory 'server.stdout.log') -RedirectStandardError (Join-Path $RuntimeDirectory 'server.stderr.log')
    $process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
}
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($process.HasExited) { throw "Ollama exited. Read $RuntimeDirectory\server.stderr.log" }
    try {
        $service = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 1
        Write-Host "Ollama running (PID $($process.Id), version $($service.version))."
        return
    } catch { Start-Sleep -Milliseconds 500 }
}
throw "Ollama did not become ready. Read $RuntimeDirectory\server.stderr.log"
