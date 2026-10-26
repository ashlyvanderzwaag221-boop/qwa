$taskStage = $PSScriptRoot
. (Join-Path (Split-Path -Parent $taskStage) '00_公共支持/runtime.ps1')
Invoke-ProjectPython -Arguments @((Join-Path $taskStage 'experiment.py'))
Write-Host 'Stage 2 complete: fitted models, validation predictions and search records.'
