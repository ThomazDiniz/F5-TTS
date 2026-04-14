@echo off
setlocal EnableDelayedExpansion
REM ============================================================================
REM Treino F5-TTS longo: dataset data\all_char (nome logico "all"), tokenizer char,
REM inicio com PESOS ALEATORIOS (sem --finetune / sem checkpoint base).
REM
REM Curvas de perda e relatorio de experimento (mestrado):
REM   - Durante: TensorBoard -> runs\%EXP_NAME%  (loss, timing/sec_per_iter)
REM   - Pasta ckpts\%DATASET_NAME%\experiment_report\
REM       system_info.json, run_config.json, dataset_estimate.json
REM       step_timing.csv (tempo por iteracao), epoch_timing.csv (tempo por epoca)
REM       time_estimate.json (atualizado apos cada epoca; estimativa tempo restante)
REM       loss_by_step.csv, loss_by_epoch.csv, statistics_summary.json
REM       loss.png, loss_by_epoch.png (copia dos graphics/)
REM   - Opcional: pip install psutil  (RAM/CPU extra em system_info.json)
REM
REM Estimativa de epocas para ~1 semana de GPU continua:
REM   epocas ~= (7 * 24 * 3600) / (segundos por epoca)
REM   Ex.: se 1 epoca demora ~2 h -> ~84 epocas em 7 dias. Faca um teste com EPOCHS=1 e cronometre.
REM ============================================================================

cd /d "%~dp0"

set "F5_CONDA_ENV=f5-tts"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

REM Dataset: pasta fisica data\all_char  =>  --dataset_name all
set "DATASET_NAME=all"
set "EXP_NAME=F5TTS_Base_all_char_long"

REM 100 epocas (ajuste conforme estimativa acima)
set "EPOCHS=100"

REM Batch em frames (reduza se OOM: 1200, 800)
set "BATCH_FRAMES=1600"
set "MAX_SAMPLES=64"
set "GRAD_ACCUM=1"

REM LR treino do zero; baixe para 5e-5 se divergir
set "LR=0.0001"
set "WARMUP=2000"

REM Checkpoints: guardar a cada N epocas + model_last a cada LAST_PER_STEPS
set "SAVE_EVERY_EPOCHS=2"
set "SAVE_PER_UPDATES=2147483647"
set "LAST_PER_STEPS=5000"

REM Manter no maximo N ficheiros model_<step>.pt (rotação); model_last.pt nunca apagado
set "CHECKPOINT_MAX_KEEP=10"

REM 1 = apaga ckpts\all\*.pt antes de comecar (treino novo do zero)
set "CLEAN_START=0"

REM 0 = gera ref.wav / gen.wav quando houver save (last_per_steps / save_every_epochs)
set "NO_LOG_SAMPLES=0"

REM Audio ref vs inferido a cada N epocas (independente do intervalo de checkpoint)
set "LOG_SAMPLES_EVERY_EPOCHS=2"

REM Registos por passo: 1 = todos (ficheiros grandes); 5 ou 10 em corridas de semanas
set "EXP_LOG_EVERY=5"

if not exist "data\all_char" (
    echo ERRO: Pasta nao encontrada: data\all_char
    echo Precisa de raw.arrow ^(ou raw^), duration.json, vocab.txt preparados.
    pause
    exit /b 1
)

if not exist "ckpts" mkdir "ckpts"
if not exist "ckpts\%DATASET_NAME%" mkdir "ckpts\%DATASET_NAME%"
if not exist "runs" mkdir "runs"

if "%CLEAN_START%"=="1" (
    if exist "ckpts\%DATASET_NAME%\*.pt" (
        echo A limpar checkpoints em ckpts\%DATASET_NAME%\ ...
        del /q "ckpts\%DATASET_NAME%\*.pt" 2>nul
    )
)

call conda activate %F5_CONDA_ENV%
if errorlevel 1 (
    echo ERRO: conda activate %F5_CONDA_ENV%
    pause
    exit /b 1
)

set "NOFLAG="
if "%NO_LOG_SAMPLES%"=="1" set "NOFLAG=--no_log_samples"

echo.
echo Dataset: data\all_char  dataset_name=%DATASET_NAME%
echo Epocas: %EPOCHS%  LR=%LR%  batch_frames=%BATCH_FRAMES%
echo Checkpoints: a cada %SAVE_EVERY_EPOCHS% epocas ^(rotacao: %CHECKPOINT_MAX_KEEP% x model_STEP.pt^) + model_last a cada %LAST_PER_STEPS% steps
echo TensorBoard: tensorboard --logdir runs --port 6006
echo Graficos + CSV: ckpts\%DATASET_NAME%\graphics\ e experiment_report\
echo.

python -u -m f5_tts.train.finetune_cli ^
  --dataset_name %DATASET_NAME% ^
  --tokenizer char ^
  --exp_name %EXP_NAME% ^
  --epochs %EPOCHS% ^
  --learning_rate %LR% ^
  --batch_size_per_gpu %BATCH_FRAMES% ^
  --batch_size_type frame ^
  --max_samples %MAX_SAMPLES% ^
  --grad_accumulation_steps %GRAD_ACCUM% ^
  --num_warmup_updates %WARMUP% ^
  --save_per_updates %SAVE_PER_UPDATES% ^
  --last_per_steps %LAST_PER_STEPS% ^
  --save_every_epochs %SAVE_EVERY_EPOCHS% ^
  --logger tensorboard ^
  --experiment_log_every_n %EXP_LOG_EVERY% ^
  --log_samples_every_n_epochs %LOG_SAMPLES_EVERY_EPOCHS% ^
  --checkpoint_max_keep %CHECKPOINT_MAX_KEEP% ^
  %NOFLAG%

set "RC=%errorlevel%"
echo.
if "%RC%" neq "0" ( echo Terminou com codigo %RC%. ) else ( echo Treino concluido. Ver ckpts\%DATASET_NAME%\experiment_report\ )
pause
exit /b %RC%
