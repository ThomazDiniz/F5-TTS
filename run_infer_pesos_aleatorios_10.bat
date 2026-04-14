@echo off
setlocal
REM Gera 10 WAVs com ckpts\pesos_aleatorios\model_last.pt
REM Requer Docker (GPU) e imagem firstpixel-f5tts:local (build_docker.bat)
cd /d "%~dp0"

if not exist "ckpts\pesos_aleatorios\model_last.pt" (
    echo ERRO: Checkpoint nao encontrado: ckpts\pesos_aleatorios\model_last.pt
    exit /b 1
)

echo [1/2] Build da imagem...
call "%~dp0build_docker.bat" nopause
if errorlevel 1 exit /b 1

if not exist "infer_out_pesos_aleatorios" mkdir "infer_out_pesos_aleatorios"

echo [2/2] Inferencia ^(5 texto aleatorio + 5 texto treino^) + manifest.csv ...
docker run --rm --gpus all ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  -v "%cd%\data:/workspace/F5-TTS/data" ^
  -v "%cd%\tools:/workspace/F5-TTS/tools" ^
  -v "%cd%\infer_out_pesos_aleatorios:/workspace/F5-TTS/infer_out_pesos_aleatorios" ^
  firstpixel-f5tts:local ^
  python /workspace/F5-TTS/tools/infer_pesos_aleatorios_batch.py

if errorlevel 1 (
    echo Falha na inferencia.
    exit /b 1
)
echo.
echo Pronto. Ouvir em: %cd%\infer_out_pesos_aleatorios\
endlocal
