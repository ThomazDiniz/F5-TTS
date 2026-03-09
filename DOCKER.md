# Rodar F5-TTS com Docker (firstpixel-f5tts:local)

## Build da imagem

Na pasta do projeto (PowerShell ou CMD):

```bash
docker build -t firstpixel-f5tts:local .
```

---

## Inferência (TTS)

Interface Gradio em **http://localhost:7860**

Cache do Hugging Face e checkpoints no host (use `%cd%` no CMD para a pasta atual):

```bash
docker run --rm -it --gpus all -p 7860:7860 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v %cd%\ckpts:/workspace/F5-TTS/ckpts ^
  firstpixel-f5tts:local ^
  f5-tts_infer-gradio --host 0.0.0.0 --port 7860
```

**Checkpoints esperados no host (ex.: projeto firstpixelptbr):**

- `ckpts/firstpixelptbr/model_last.safetensors`
- `ckpts/firstpixelptbr/model_200000.pt`

Dentro do container esses arquivos ficam em:

- `/workspace/F5-TTS/ckpts/firstpixelptbr/model_last.safetensors`
- `/workspace/F5-TTS/ckpts/firstpixelptbr/model_200000.pt`

---

## Finetune

Interface Gradio em **http://localhost:7861** (porta diferente para não conflitar com a inferência)

```bash
docker run --rm -it --gpus all -p 7861:7860 ^
  -v f5tts_hf_cache:/root/.cache/huggingface ^
  -v %cd%\ckpts:/workspace/F5-TTS/ckpts ^
  -v %cd%\data:/workspace/F5-TTS/data ^
  firstpixel-f5tts:local ^
  f5-tts_finetune-gradio --host 0.0.0.0 --port 7860
```

**Checkpoints (ex.: firstpixelptbr):**

- `/workspace/F5-TTS/ckpts/firstpixelptbr/model_200000.pt`
- `/workspace/F5-TTS/ckpts/firstpixelptbr/model_last.pt`

---

## Shell dentro do container

```bash
docker run --rm -it firstpixel-f5tts:local bash
```

Útil para testar CLI, inspecionar arquivos em `/workspace/F5-TTS/ckpts` e `/workspace/F5-TTS/data`, etc.

---

## Resumo

| Modo       | Porta no host | URL                    | Volumes |
|-----------|----------------|------------------------|--------|
| Inferência | 7860           | http://localhost:7860  | HF cache + `ckpts` |
| Finetune   | 7861           | http://localhost:7861  | HF cache + `ckpts` + `data` |

**Imagem:** `firstpixel-f5tts:local`  
**Volume nomeado (cache):** `f5tts_hf_cache` (persiste entre runs)
