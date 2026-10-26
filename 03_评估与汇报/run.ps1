$taskStage = $PSScriptRoot
. (Join-Path (Split-Path -Parent $taskStage) '00_公共支持/runtime.ps1')
Invoke-ProjectPython -Arguments @('-m', 'unittest', 'discover', '-s', (Join-Path $taskStage 'tests'), '-v')
Invoke-ProjectPython -Arguments @((Join-Path $taskStage 'analyze_results.py'))
Invoke-ProjectPython -Arguments @((Join-Path $taskStage 'verify_results.py'))
Write-Host 'Stage 3 complete: evaluation figures and saved-result verification. Report and slides remain available in reports.'
