@echo off
REM F5-TTS: Rodar interface de finetune
cd /d "%~dp0"
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" f5-tts
f5-tts_finetune-gradio
pause
