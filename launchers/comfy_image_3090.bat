@echo off
REM ============================================================
REM  ComfyUI - harness IMAGE instance (Krea 2) on the RTX 3090
REM ============================================================
REM  Port 8190, so it never collides with the Stability Matrix
REM  instance on 8188. CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA
REM  number GPUs like nvidia-smi does:
REM    0 = 5090   1 = 4090   2 = 3090
REM  models/, custom_nodes/, input/ and output/ are shared with the
REM  other instances; user dir and SQLite DB are separate so the
REM  processes never lock each other's database.
REM  No --use-sage-attention: on the 3090 it renders black images
REM  (NaNs) with Krea 2, measured 2026-10-13. It works on the 5090.
REM ============================================================
setlocal
set "COMFY_DIR=H:\Stability Matrix\StabilityMatrix-win-x64\Data\Packages\ComfyUI"
set "PORT=8190"
set "CUDA_DEV=2"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"

if not exist "%COMFY_DIR%\main.py" (
    echo ERROR: ComfyUI not found at %COMFY_DIR%
    pause
    exit /b 1
)
cd /d "%COMFY_DIR%"
if not exist "user_harness_image" mkdir "user_harness_image"
title ComfyUI harness IMAGE - port %PORT% - RTX 3090
echo Starting ComfyUI IMAGE instance on port %PORT%, CUDA device %CUDA_DEV% (expect RTX 3090 below)
venv\Scripts\python.exe main.py ^
    --port %PORT% ^
    --cuda-device %CUDA_DEV% ^
    --bf16-text-enc ^
    --user-directory "%COMFY_DIR%\user_harness_image" ^
    --database-url "sqlite:///user_harness_image/comfyui_harness_image.db"
endlocal
pause
