@echo off
echo =======================================================
echo    EvihAnimation x MotionBricks: Random Walk Demo
echo =======================================================
echo Initializing PyTorch Neural Inference and PyRay...
echo Please wait for the window to appear (may take 10-20 seconds).
echo The robot will autonomously walk and turn using neural control.

wsl -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate motionbricks && cd /mnt/d/AnimationTech-learning/EvihAnimation-motionbricks-replay/Demos/MotionBricksInteractive && env LD_PRELOAD=/root/miniconda3/envs/motionbricks/lib/libstdc++.so.6 python InteractiveMain.py --controller random"

echo.
echo Demo finished.
pause
