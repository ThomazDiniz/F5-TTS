@echo off
REM Inferencia - firstpixel-f5tts:local -> http://localhost:7860
REM Workspace: ckpts do host montado em /workspace/F5-TTS/ckpts (ex.: firstpixelptbr/model_last.safetensors)
cd /d "%~dp0"
if not exist "ckpts" mkdir ckpts
docker run --rm -it --gpus all -p 7860:7860 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  firstpixel-f5tts:local ^
  f5-tts_infer-gradio --host 0.0.0.0 --port 7860
pause
