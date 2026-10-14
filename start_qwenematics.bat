@echo off
REM ============================================================
REM  Qwenematics - starts the GUI server and opens it.
REM  ComfyUI instances and Ninfer are launched from the GUI's
REM  Settings tab (or launchers\*.bat / launch-ninfer-4090.bat).
REM ============================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment...
    py -3.13 -m venv .venv || goto :fail
    .venv\Scripts\python -m pip install -r requirements.txt || goto :fail
)
if not exist "frontend\dist\index.html" (
    echo Building the UI...
    pushd frontend && call npm install && call npm run build && popd
)
title Qwenematics - http://127.0.0.1:8765
.venv\Scripts\python -m harness.server --open
if errorlevel 1 pause
exit /b
:fail
echo Setup failed.
pause
