@echo off
echo =======================================================
echo    EvihAnimation x MotionBricks: Native Windows WASD Demo
echo =======================================================
echo Initializing PyTorch Neural Inference (CPU Fallback) and PyRay...
echo Please wait for the window to appear.
echo Use W, A, S, D to control the robot.

cd Demos\MotionBricksInteractive
python InteractiveMain.py --controller wasd
cd ..\..

echo.
echo Demo finished.
pause
