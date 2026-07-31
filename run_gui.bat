@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment was not found.
    echo Run setup.ps1 first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m hardsub_ocr.app
if errorlevel 1 pause
