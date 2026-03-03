@echo off
REM Build da imagem firstpixel-f5tts:local
cd /d "%~dp0"
docker build -t firstpixel-f5tts:local .
echo.
echo Build concluido. Use run_docker_inference.bat ou run_docker_finetune.bat para rodar.
pause
