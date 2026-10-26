param([switch]$Download)
$taskStage = $PSScriptRoot
. (Join-Path (Split-Path -Parent $taskStage) '00_公共支持/runtime.ps1')
if ($Download) {
    Invoke-ProjectPython -Arguments @((Join-Path $taskStage 'download_dataset.py'))
}
Invoke-ProjectPython -Arguments @((Join-Path $taskStage 'validate_metadata.py'))
Invoke-ProjectPython -Arguments @((Join-Path $taskStage 'extract_features.py'))
Write-Host 'Stage 1 complete: verified metadata and 8732 x 40 features.'
