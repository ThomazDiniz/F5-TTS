# Checkpoints

Pretrained model ckpts: https://huggingface.co/SWivid/F5-TTS

```
ckpts/
    E2TTS_Base/
        model_1200000.pt
    F5TTS_Base/
        model_1200000.pt
```

## firstpixelptbr (base PT-BR)

Checkpoint base em português do Brasil para finetune. **Obrigatório** para treinar a partir dele:

- `ckpts/firstpixelptbr/model_last.pt` (ou `model_XXXXX.pt`)
- `ckpts/firstpixelptbr/vocab.txt` — mesmo vocabulário com que o modelo foi treinado (uma linha por caractere; espaço deve ser índice 0)

Ao usar firstpixelptbr como base, o **vocab do seu projeto de treino tem de ser idêntico** a esse `vocab.txt` (mesmos caracteres, mesma ordem). Ver [FIRSTPIXELPTBR.md](FIRSTPIXELPTBR.md) na raiz do projeto.

**Onde achar um vocab para firstpixelptbr:** O firstpixel/F5-TTS-pt-br no Hugging Face não publica `vocab.txt`; o modelo usa o tokenizer do F5-TTS (SWivid). Você pode copiar o vocab do próprio projeto: `src/f5_tts/infer/examples/vocab.txt` → `ckpts/firstpixelptbr/vocab.txt`, ou baixar o [vocab.txt do SWivid/F5-TTS F5TTS_Base](https://huggingface.co/SWivid/F5-TTS/blob/main/F5TTS_Base/vocab.txt).