@echo off
setlocal EnableExtensions

REM ==========================================================
REM One-click: cria env conda, instala deps e abre finetune UI
REM ==========================================================

cd /d "%~dp0"

set "F5_CONDA_ENV=f5-tts"
set "F5_PYTHON_VERSION=3.10"
set "F5_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
set "F5_FINETUNE_HOST=127.0.0.1"
set "F5_FINETUNE_PORT=7861"
set "F5_BOOTSTRAP_MARKER=.f5_bootstrap_done"
set "F5_VERBOSE=1"

echo.
echo [0/6] Preparando diretorio...
if not exist "data" (
    echo ERRO: Pasta "data" nao encontrada em:
    echo   %cd%\data
    echo Crie/preencha a pasta data antes de abrir o Gradio de treino.
    pause
    exit /b 1
)
if not exist "ckpts" mkdir "ckpts"

echo.
echo [1/6] Localizando Conda...
if "%F5_VERBOSE%"=="1" echo [DEBUG] Procurando conda.bat nos locais padrao e no PATH...
set "CONDA_BAT="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT (
    for /f "delims=" %%i in ('where conda.bat 2^>nul') do (
        set "CONDA_BAT=%%i"
        goto :conda_found
    )
)
:conda_found
if not defined CONDA_BAT (
    echo ERRO: conda.bat nao encontrado.
    echo Instale Miniconda/Anaconda e rode novamente.
    pause
    exit /b 1
)
echo Conda encontrado em: %CONDA_BAT%

echo.
echo [2/6] Checando ambiente "%F5_CONDA_ENV%"...
if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda env list ^| findstr ...
call "%CONDA_BAT%" env list | findstr /R /C:"^[ ]*%F5_CONDA_ENV%[ ]" >nul
if errorlevel 1 (
    echo Ambiente nao existe. Criando...
    call "%CONDA_BAT%" create -n "%F5_CONDA_ENV%" -y python=%F5_PYTHON_VERSION%
    if errorlevel 1 (
        echo ERRO: falha ao criar ambiente conda.
        pause
        exit /b 1
    )
) else (
    echo Ambiente ja existe.
)

echo.
echo [3/6] Ativando ambiente...
if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python --version
call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python --version
if errorlevel 1 (
    echo ERRO: ambiente "%F5_CONDA_ENV%" nao esta operacional.
    pause
    exit /b 1
)

echo.
echo [4/6] Instalando/atualizando dependencias [Conda + pip]...

if not exist "%F5_BOOTSTRAP_MARKER%" (
    echo Instalando base com pip dentro do ambiente Conda...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install --upgrade pip setuptools wheel
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 (
        echo ERRO: falha ao atualizar ferramentas base do pip.
        pause
        exit /b 1
    )

    echo Instalando PyTorch/Torchaudio via pip ^(CUDA 12.4^)...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url %F5_TORCH_INDEX_URL%
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url %F5_TORCH_INDEX_URL%
    if errorlevel 1 (
        echo AVISO: falha no wheel CUDA. Tentando fallback CPU via PyPI...
        call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install torch==2.4.0 torchaudio==2.4.0
        if errorlevel 1 (
            echo ERRO: falha ao instalar pytorch/torchaudio via pip.
            pause
            exit /b 1
        )
    )

    echo Instalando libs de audio sem stack GTK...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install soundfile imageio-ffmpeg
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install soundfile imageio-ffmpeg
    if errorlevel 1 (
        echo ERRO: falha ao instalar libs de audio.
        pause
        exit /b 1
    )

    echo Instalando projeto em modo editavel...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install -e .
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install -e .
    if errorlevel 1 (
        echo ERRO: falha ao instalar dependencias do projeto.
        pause
        exit /b 1
    )

    echo Ajustando stack web compativel com Gradio 3.50.x...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install fastapi==0.103.2 starlette==0.27.0 jinja2==3.1.4 pydantic^<2.0.0
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install "fastapi==0.103.2" "starlette==0.27.0" "jinja2==3.1.4" "pydantic<2.0.0"
    if errorlevel 1 (
        echo ERRO: falha ao ajustar stack web para Gradio 3.x.
        pause
        exit /b 1
    )

    echo ok>"%F5_BOOTSTRAP_MARKER%"
) else (
    echo Bootstrap inicial ja concluido anteriormente. Rodando update leve do projeto...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install -e .
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install -e .
    if errorlevel 1 (
        echo ERRO: falha ao atualizar instalacao do projeto.
        pause
        exit /b 1
    )

    echo Reaplicando stack web compativel com Gradio 3.50.x...
    if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -m pip install fastapi==0.103.2 starlette==0.27.0 jinja2==3.1.4 pydantic^<2.0.0
    call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -m pip install "fastapi==0.103.2" "starlette==0.27.0" "jinja2==3.1.4" "pydantic<2.0.0"
    if errorlevel 1 (
        echo ERRO: falha ao ajustar stack web para Gradio 3.x.
        pause
        exit /b 1
    )
)

echo.
echo [5/6] Validando comando f5-tts_finetune-gradio...
if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% python -c "import torch, torchaudio; ..."
call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" python -c "import torch, torchaudio; print('torch=', torch.__version__, 'cuda=', torch.version.cuda); print('torchaudio=', torchaudio.__version__)"
if errorlevel 1 (
    echo ERRO: stack torch/torchaudio invalida no ambiente.
    pause
    exit /b 1
)

echo.
echo [6/6] Abrindo Gradio de treino:
echo   http://localhost:%F5_FINETUNE_PORT%
echo.
echo Dica: cada treino vai gerar train.log em ckpts\SEU_PROJETO\train.log
echo.
if "%F5_VERBOSE%"=="1" echo [DEBUG] Comando: conda run -n %F5_CONDA_ENV% f5-tts_finetune-gradio --host %F5_FINETUNE_HOST% --port %F5_FINETUNE_PORT%
call "%CONDA_BAT%" run -n "%F5_CONDA_ENV%" f5-tts_finetune-gradio --host %F5_FINETUNE_HOST% --port %F5_FINETUNE_PORT%

set "RC=%errorlevel%"
echo.
if "%RC%" neq "0" (
    echo Finetune Gradio terminou com erro ^(codigo %RC%^).
) else (
    echo Finetune Gradio encerrado.
)
pause
exit /b %RC%
