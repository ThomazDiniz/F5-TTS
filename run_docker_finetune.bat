@echo off
REM Finetune - firstpixel-f5tts:local -> http://localhost:7861
REM Sempre "fast": NAO roda docker build. Imagem: build_docker.bat (uma vez) ou manual.
cd /d "%~dp0"

if not exist "ckpts" mkdir ckpts
if not exist "data" mkdir data

echo Gradio finetune ^(imagem: firstpixel-f5tts:local, codigo: volume^)...
docker run --rm -it --gpus all -p 7861:7860 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v "%cd%:/workspace/F5-TTS" ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  -v "%cd%\data:/workspace/F5-TTS/data" ^
  firstpixel-f5tts:local ^
  f5-tts_finetune-gradio --host 0.0.0.0 --port 7860

pause
