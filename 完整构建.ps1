param(
    [string]$Python = 'python',
    [ValidateSet('cached', 'full')][string]$Mode = 'cached',
    [string]$Archive = '',
    [switch]$Download,
    [int]$Workers = 4
)
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}
Invoke-Checked $Python @((Join-Path $projectDirectory 'scripts\bootstrap.py'))
$environmentPython = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$arguments = @((Join-Path $projectDirectory 'scripts\reproduce.py'), '--mode', $Mode, '--workers', "$Workers")
if ($Archive) { $arguments += @('--archive', $Archive) }
if ($Download) { $arguments += '--download' }
Invoke-Checked $environmentPython $arguments
Write-Host 'Reproduction and numerical verification passed. See reproduced/verification.json.'
