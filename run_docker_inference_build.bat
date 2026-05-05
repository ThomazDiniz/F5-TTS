@echo off
REM Inferencia com build - firstpixel-f5tts:local -> http://localhost:7860
cd /d "%~dp0"

echo.
echo [1/2] Build image...
call "%~dp0build_docker.bat" nopause
if errorlevel 1 (
    echo.
    echo Build falhou. Abortando run.
    pause
    exit /b 1
)

echo.
echo [2/2] Run inference...
call "%~dp0run_docker_inference.bat"
