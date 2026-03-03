@echo off
setlocal EnableDelayedExpansion
title F5-TTS - Instalacao
cd /d "%~dp0"

echo.
echo ============================================
echo   F5-TTS - Instalacao do ambiente
echo ============================================
echo.

REM Encontrar Conda (Miniconda ou Anaconda)
set "CONDA_ROOT="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.bat" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if exist "%USERPROFILE%\anaconda3\Scripts\conda.bat" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if exist "C:\ProgramData\miniconda3\Scripts\conda.bat" set "CONDA_ROOT=C:\ProgramData\miniconda3"
if exist "C:\ProgramData\anaconda3\Scripts\conda.bat" set "CONDA_ROOT=C:\ProgramData\anaconda3"

if "%CONDA_ROOT%"=="" (
    echo [ERRO] Conda nao encontrado. Instale Miniconda ou Anaconda:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)
echo [OK] Conda: %CONDA_ROOT%
set "PATH=%CONDA_ROOT%;%CONDA_ROOT%\Scripts;%CONDA_ROOT%\Library\bin;%PATH%"
echo.

REM Remover ambiente antigo se existir
echo [1/4] Verificando ambiente existente...
conda env list | findstr /C:"f5-tts " >nul 2>&1
if %errorlevel%==0 (
    echo       Ambiente f5-tts existe. Removendo para reinstalar...
    call conda env remove -n f5-tts -y
    echo.
)

REM Criar ambiente a partir do environment.yml
echo [2/4] Criando ambiente conda f5-tts (Python 3.10)...
call conda env create -f environment.yml -y
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao criar ambiente.
    pause
    exit /b 1
)
echo.

REM Instalar PyTorch com CUDA 11.8 (compativel com transformers e F5-TTS)
echo [3/4] Instalando PyTorch 2.4 + CUDA 11.8...
call conda run -n f5-tts pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar PyTorch.
    pause
    exit /b 1
)
echo.

REM Instalar o projeto e dependencias
echo [4/4] Instalando F5-TTS e dependencias...
call conda run -n f5-tts pip install -e .
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar o projeto.
    pause
    exit /b 1
)
echo.

echo ============================================
echo   Instalacao concluida.
echo ============================================
echo.
echo Para rodar:
echo   - Inferencia (TTS): run_inference.bat  ou  run_project.bat
echo   - Finetune:         run_finetune.bat
echo.
echo Ou no terminal: conda activate f5-tts
echo                 f5-tts_infer-gradio   /   f5-tts_finetune-gradio
echo.
pause
