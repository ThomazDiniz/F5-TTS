@echo off
REM Inferencia - firstpixel-f5tts:local -> http://localhost:7860
REM Workspace: ckpts do host montado em /workspace/F5-TTS/ckpts (ex.: firstpixelptbr/model_last.safetensors)
cd /d "%~dp0"

echo.
echo [Run] Inference (sem build automatico)...
echo Dica: rode build_docker.bat manualmente quando mudar Dockerfile/dependencias.
if not exist "ckpts" mkdir ckpts
if not exist "data" mkdir data
docker run --rm -it --gpus all -p 7860:7860 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v "%cd%:/workspace/F5-TTS" ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  -v "%cd%\data:/workspace/F5-TTS/data" ^
  firstpixel-f5tts:local ^
  python /workspace/F5-TTS/src/f5_tts/infer/infer_gradio.py --host 0.0.0.0 --port 7860
pause
