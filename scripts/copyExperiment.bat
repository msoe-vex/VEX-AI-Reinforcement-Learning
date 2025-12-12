@echo off
setlocal enabledelayedexpansion

set HOST=ROSIE
set trialDirectory=/home/ad.msoe.edu/needhama/Documents/GitHub/VEX-AI-Reinforcement-Learning/job_results/job_215219/PPO_2025-12-12_00-47-54
for %%F in ("%trialDirectory%") do set "lastFolder=%%~nxF"

scp -r "%HOST%:%trialDirectory%/" "%cd%\vex_model_training\%lastFolder%"

pause