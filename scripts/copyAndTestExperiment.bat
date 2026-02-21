@echo off
setlocal enabledelayedexpansion

set HOST=ROSIE

set /p trialDirectory=Enter the full path to the trial directory on %HOST%: 
for %%F in ("%trialDirectory%") do set "lastFolder=%%~nxF"

scp -r "%HOST%:%trialDirectory%/" "%cd%\vex_model_training\%lastFolder%"

if %errorlevel% neq 0 (
    echo Error: Failed to copy directory from %HOST%.
    exit /b %errorlevel%
)

python vex_model_test.py --experiment-path "vex_model_training\%lastFolder%"