@echo off
setlocal enabledelayedexpansion

set HOST=ROSIE
set trialDirectory=/home/ad.msoe.edu/needhama/Documents/GitHub/VEX-AI-Reinforcement-Learning/job_results/job_215268/PPO_2025-12-12_09-44-33
for %%F in ("%trialDirectory%") do set "lastFolder=%%~nxF"

scp -r "%HOST%:%trialDirectory%/" "%cd%\vex_model_training\%lastFolder%"

pause