@echo off
setlocal EnableDelayedExpansion
REM Treino no Docker (firstpixel-f5tts:local): pesos aleatorios, dataset data\pesos_aleatorios_char
REM TensorBoard roda DENTRO do container (tensorboard ja esta no Dockerfile) -> http://localhost:6006
REM Logs ficam em runs\ no host (volume montado) -> runs\F5TTS_Base, escalar "loss"

cd /d "%~dp0"

REM 1 = apaga checkpoints .pt em ckpts\pesos_aleatorios (treino do zero, sem retomar)
set "CLEAN_START=1"

REM 1 = docker build --no-cache (bem mais lento; use se a imagem estiver corrompida ou quiser pip de novo)
set "DOCKER_BUILD_NOCACHE=0"

set "BATCH_FRAMES=1600"
set "EPOCHS=5"
set "SAVE_EVERY=200"
set "WARMUP=100"
set "LR=0.0001"

if not exist "data\pesos_aleatorios_char" (
    echo ERRO: Pasta nao encontrada: data\pesos_aleatorios_char
    echo Prepare os dados ^(raw ou raw.arrow, duration.json, vocab.txt^).
    pause
    exit /b 1
)

if "%CLEAN_START%"=="1" (
    if exist "ckpts\pesos_aleatorios\*.pt" (
        echo Limpando checkpoints antigos em ckpts\pesos_aleatorios ...
        del /q "ckpts\pesos_aleatorios\*.pt" 2>nul
    )
)

if not exist "ckpts" mkdir "ckpts"
if not exist "data" mkdir "data"
if not exist "ckpts\pesos_aleatorios" mkdir "ckpts\pesos_aleatorios"
if not exist "runs" mkdir "runs"

echo.
echo [1/2] Rebuild da imagem Docker ^(sempre antes do treino^)...
if "%DOCKER_BUILD_NOCACHE%"=="1" (
    docker build --no-cache -t firstpixel-f5tts:local .
) else (
    docker build -t firstpixel-f5tts:local .
)
if errorlevel 1 (
    echo Rebuild falhou. Abortando.
    echo Se aparecer erro de cache do Docker, use: build_docker.bat clean
    pause
    exit /b 1
)

echo.
echo [2/2] Treino + TensorBoard no container...
echo Abra no navegador: http://localhost:6006
echo   TensorBoard: escalar "loss" ^(eixo X = step global^) e "loss/epoch_mean" ^(eixo X = numero da epoca^)
echo   Logs: %cd%\runs\F5TTS_Base
echo   PNG ao fim do treino: %cd%\ckpts\pesos_aleatorios\graphics\loss.png e loss_by_epoch.png
echo.

docker run --rm -it --gpus all -p 6006:6006 ^
  -v "%cd%\ckpts:/workspace/F5-TTS/ckpts" ^
  -v "%cd%\data:/workspace/F5-TTS/data" ^
  -v "%cd%\runs:/workspace/F5-TTS/runs" ^
  firstpixel-f5tts:local ^
  bash -lc "tensorboard --logdir /workspace/F5-TTS/runs --host 0.0.0.0 --port 6006 & exec f5-tts_finetune-cli --dataset_name pesos_aleatorios --tokenizer char --exp_name F5TTS_Base --epochs %EPOCHS% --learning_rate %LR% --batch_size_per_gpu %BATCH_FRAMES% --batch_size_type frame --max_samples 64 --num_warmup_updates %WARMUP% --save_per_updates %SAVE_EVERY% --last_per_steps %SAVE_EVERY% --save_every_epochs 0 --logger tensorboard --no_log_samples"

echo.
if errorlevel 1 (
    echo Container terminou com erro ^(codigo %errorlevel%^).
) else (
    echo Treino finalizado.
)
pause
endlocal
