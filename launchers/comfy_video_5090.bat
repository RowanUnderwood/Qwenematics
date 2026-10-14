@echo off
REM ============================================================
REM  ComfyUI - harness VIDEO instance (MiniMax H3) on the RTX 5090
REM ============================================================
REM  Same as "H:\Image and Video extender\run_comfy_video.bat"
REM  (port 8189, own user dir and DB) except that it sets
REM  CUDA_DEVICE_ORDER=PCI_BUS_ID, so device 0 is the 5090 here,
REM  matching nvidia-smi:  0 = 5090   1 = 4090   2 = 3090
REM  Do not run both this and run_comfy_video.bat - same port.
REM ============================================================
setlocal
set "COMFY_DIR=H:\Stability Matrix\StabilityMatrix-win-x64\Data\Packages\ComfyUI"
set "PORT=8189"
set "CUDA_DEV=0"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"

if not exist "%COMFY_DIR%\main.py" (
    echo ERROR: ComfyUI not found at %COMFY_DIR%
    pause
    exit /b 1
)
cd /d "%COMFY_DIR%"
if not exist "user_video" mkdir "user_video"
title ComfyUI harness VIDEO - port %PORT% - RTX 5090
echo Starting ComfyUI VIDEO instance on port %PORT%, CUDA device %CUDA_DEV% (expect RTX 5090 below)
venv\Scripts\python.exe main.py ^
    --port %PORT% ^
    --cuda-device %CUDA_DEV% ^
    --use-sage-attention ^
    --reserve-vram 0.9 ^
    --user-directory "%COMFY_DIR%\user_video" ^
    --database-url "sqlite:///user_video/comfyui_video.db"
endlocal
pause
