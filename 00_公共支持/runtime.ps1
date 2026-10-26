$ErrorActionPreference = 'Stop'
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $script:ProjectRoot
$env:PYTHONUTF8 = '1'
$env:NUMBA_CACHE_DIR = Join-Path $script:ProjectRoot '01_数据处理与特征/features/numba'
$env:MPLCONFIGDIR = Join-Path $script:ProjectRoot '01_数据处理与特征/features/matplotlib'
$env:OMP_NUM_THREADS = '1'
$script:ProjectPython = Join-Path $script:ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $script:ProjectPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required to create the environment.' }
    & $script:ProjectPython -m pip install -r (Join-Path $script:ProjectRoot '00_公共支持/requirements-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $script:ProjectPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Project step failed: $Arguments" }
}
