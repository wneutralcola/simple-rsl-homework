$ErrorActionPreference = 'Stop'

function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())
    try {
        & $Executable @Arguments -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

$pythonPath = $env:RSL_PYTHON
$pythonArguments = @()
if ($pythonPath) {
    if (-not (Test-PythonRuntime -Executable $pythonPath)) {
        throw 'RSL_PYTHON must point to a working Python 3.10+ executable.'
    }
} else {
    # Prefer a normal Python installation; a Codex runtime is only an optional fallback.
    $candidates = @(
        @{ Name = 'python'; Arguments = @() },
        @{ Name = 'python3'; Arguments = @() },
        @{ Name = 'py'; Arguments = @('-3') }
    )
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        # Windows app execution aliases can open the Store instead of running Python.
        if ($command -and $command.Source -notmatch '\\Microsoft\\WindowsApps\\' -and
            (Test-PythonRuntime -Executable $command.Source -Arguments $candidate.Arguments)) {
            $pythonPath = $command.Source
            $pythonArguments = $candidate.Arguments
            break
        }
    }
}
if (-not $pythonPath -and $env:USERPROFILE) {
    $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if ((Test-Path -LiteralPath $bundled) -and (Test-PythonRuntime -Executable $bundled)) {
        $pythonPath = $bundled
    }
}
if (-not $pythonPath) { throw 'Install Python 3.10+ or set RSL_PYTHON to its executable path.' }
$env:PYTHONUTF8 = '1'
& $pythonPath @pythonArguments (Join-Path $PSScriptRoot 'rsl.py') @args
exit $LASTEXITCODE
