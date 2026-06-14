@echo off
echo =======================================================
echo    EvihAnimation x MotionBricks: Native Windows Random Walk
echo =======================================================
echo Initializing PyTorch Neural Inference (CPU Fallback) and PyRay...
echo Please wait for the window to appear.

cd Demos\MotionBricksInteractive
python InteractiveMain.py --controller random
cd ..\..

echo.
echo Demo finished.
pause
