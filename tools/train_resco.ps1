# Train a RESCO reference agent on the Cologne scenario.
# Handles the dedicated Python 3.10 venv + SUMO env so you don't have to.
#
#   .\tools\train_resco.ps1                      # IPPO, 200 episodes
#   .\tools\train_resco.ps1 -Episodes 100        # shorter
#   .\tools\train_resco.ps1 -Agent MPLight       # (note: DQN agents crash on pfrl 0.4)
#
# Each episode is about 75 s via TraCI, so 200 episodes is roughly 4 hours.
# The best model and results are written under the RESCO results/ folder.
param(
    [string]$Agent = "IPPO",
    [int]$Episodes = 200
)

$proj  = Split-Path -Parent $PSScriptRoot
$venv  = Join-Path $proj ".venv-resco"
$vpy   = Join-Path $venv "Scripts\python.exe"
$resco = "C:\Users\tahae\AppData\Local\Temp\RESCO\resco_benchmark"

if (-not (Test-Path $vpy))   { Write-Error "venv missing at $venv (see src/realcity/RESCO_SETUP.md)"; exit 1 }
if (-not (Test-Path $resco)) { Write-Error "RESCO missing at $resco - re-clone it (see src/realcity/RESCO_SETUP.md)"; exit 1 }

# SUMO must come from the venv (system SUMO 1.26 DLLs clash with venv libsumo 1.27).
$env:SUMO_HOME = Join-Path $venv "Lib\site-packages\sumo"
$env:PATH      = "$($env:SUMO_HOME)\bin;$env:PATH"

Write-Host "Training RESCO $Agent on cologne8 for $Episodes episodes (libsumo:False / TraCI)..."
Set-Location $resco
# libsumo:False because importing pfrl breaks libsumo on Windows (see RESCO_SETUP.md).
& $vpy main.py "@cologne8" "@$Agent" libsumo:False gui:False episodes:$Episodes save_console_log:False
