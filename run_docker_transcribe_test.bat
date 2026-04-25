@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Run ASR transcribe() inside the same image + bind mounts as finetune (tests utils_infer, Whisper, ffmpeg env).
REM Usage:
REM   run_docker_transcribe_test.bat
REM   run_docker_transcribe_test.bat data\all_filtered_char\wavs\segment_0.wav
REM (path is relative to repo root; in-container: /workspace/F5-TTS/...)
cd /d "%~dp0"

if not exist "ckpts" mkdir ckpts
if not exist "data" mkdir data

set "WAV_ARG=data/all_filtered_char/wavs/segment_0.wav"
if not "%~1"=="" set "WAV_ARG=%~1"
REM Backslashes to slashes for Python on Linux
set "WAV_ARG=!WAV_ARG:\=/!"

echo.
echo === Docker transcribe test ===
echo Image: firstpixel-f5tts:local  (build with build_docker.bat if missing)
echo WAV:   %WAV_ARG%
echo.
echo First run may download openai/whisper-large-v3-turbo; can take several minutes.
echo.

docker run --rm --gpus all -e PYTHONUNBUFFERED=1 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v "%cd%:/workspace/F5-TTS" ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  -v "%cd%\data:/workspace/F5-TTS/data" ^
  -w /workspace/F5-TTS ^
  firstpixel-f5tts:local ^
  python scripts/sanity_transcribe_wav.py "%WAV_ARG%"

set "EC=%errorlevel%"
echo.
if "%EC%"=="0" (echo [OK] transcribe test finished.) else (echo [FAIL] exit code %EC%.)
exit /b %EC%
