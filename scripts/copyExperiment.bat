@echo off
setlocal enabledelayedexpansion

set HOST=ROSIE
set trialDirectory=/home/ad.msoe.edu/needhama/Documents/GitHub/VEX-AI-Reinforcement-Learning/job_results/job_215762/PPO_2025-12-13_09-34-52
for %%F in ("%trialDirectory%") do set "lastFolder=%%~nxF"

scp -r "%HOST%:%trialDirectory%/" "%cd%\vex_model_training\%lastFolder%"

pause