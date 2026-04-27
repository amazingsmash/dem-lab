@echo off
setlocal

cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"

set "PYTHON_EXE=python"
where python >nul 2>nul
if errorlevel 1 (
    set "PYTHON_EXE=C:\Users\Jose\AppData\Local\Programs\Python\Python310\python.exe"
)

echo Starting LoD Terrarium Blend Viewer on http://127.0.0.1:%PORT%/
echo.
echo Keep this window open while using the viewer.
echo Press Ctrl+C to stop the server.
echo.

"%PYTHON_EXE%" scripts\lod_terrarium_viewer.py --port %PORT% --open

endlocal
