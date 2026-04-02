@echo off
REM Build da imagem firstpixel-f5tts:local
REM O modo "clean" evita o erro "parent snapshot sha256 does not exist" (cache corrompido).
REM Uso: build_docker.bat [clean] [nopause]
cd /d "%~dp0"
set "NO_PAUSE=0"
if /i "%2"=="nopause" set "NO_PAUSE=1"
if /i "%1"=="nopause" set "NO_PAUSE=1"

if "%1"=="clean" (
    echo Limpando cache do Docker Builder...
    docker builder prune -af
    echo.
    echo Executando build sem cache...
    docker build --no-cache -t firstpixel-f5tts:local .
) else (
    docker build -t firstpixel-f5tts:local .
    if errorlevel 1 (
        echo.
        echo ERRO no build. Se apareceu "parent snapshot" ou "does not exist", execute:
        echo   build_docker.bat clean
        echo.
        if "%NO_PAUSE%"=="0" pause
        exit /b 1
    )
)
echo.
echo Build concluido. Use run_docker_inference.bat ou run_docker_finetune.bat para rodar.
if "%NO_PAUSE%"=="0" pause
