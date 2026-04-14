# Exemplo: treino F5 com logs visiveis no console E num ficheiro (tee).
# O tqdm ia para stderr por defeito; o trainer agora usa stdout. Mesmo assim,
# use 2>&1 para misturar stderr com stdout ao gravar.
#
# Uso (na raiz do repo, conda activo):
#   pwsh -File tools/train_finetune_tee.ps1
#
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot\..

$log = Join-Path (Get-Location) "train_console.log"
Write-Host "[tee] Logging to $log and console (python -u unbuffered)..."

# -u : stdout/stderr sem buffer (essencial com Tee-Object)
python -u -m f5_tts.train.finetune_cli @args 2>&1 | Tee-Object -FilePath $log
