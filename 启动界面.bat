@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
  echo Please run: python scripts\bootstrap.py
  pause
  exit /b 1
)
"%PROJECT_ROOT%.venv\Scripts\python.exe" "%PROJECT_ROOT%run_app.py"
if errorlevel 1 pause
