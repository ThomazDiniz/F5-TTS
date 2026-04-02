@echo off
REM Finetune - firstpixel-f5tts:local -> http://localhost:7861
REM Workspace: ckpts e data do host em /workspace/F5-TTS/ckpts e /workspace/F5-TTS/data
cd /d "%~dp0"

echo.
echo [1/2] Build (sempre antes do run)...
call "%~dp0build_docker.bat" nopause
if errorlevel 1 (
    echo.
    echo Build falhou. Abortando run.
    pause
    exit /b 1
)

echo.
echo [2/2] Run finetune...
if not exist "ckpts" mkdir ckpts
if not exist "data" mkdir data
docker run --rm -it --gpus all -p 7861:7860 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  -v "%cd%\data:/workspace/F5-TTS/data" ^
  firstpixel-f5tts:local ^
  f5-tts_finetune-gradio --host 0.0.0.0 --port 7860
pause
