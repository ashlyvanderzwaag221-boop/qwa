param([switch]$SkipDownload)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '01_数据处理与特征/run.ps1') -Download:(-not $SkipDownload)
& (Join-Path $PSScriptRoot '02_模型训练/run.ps1')
& (Join-Path $PSScriptRoot '03_评估与汇报/run.ps1')
Write-Host 'All three stages completed. See the responsibility folders and their READMEs.'
