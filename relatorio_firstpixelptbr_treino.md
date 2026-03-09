# Relatório: Treinar com base firstpixelptbr

**Objetivo:** Garantir que o projeto está configurado para treinar a partir do checkpoint **firstpixelptbr** sem mismatch de vocabulário e com áudio correto na inferência.

---

## 1. O que foi verificado no código

- **Carregamento do checkpoint:** O trainer carrega `ema_model_state_dict` e `model_state_dict` do `.pt` no início do treino. O `load_state_dict` é **strict** (padrão): se o tamanho da camada de embedding de texto (vocab_size) do checkpoint for diferente do modelo atual, o carregamento **falha** ou os pesos não batem.
- **Origem do vocab no treino:** Com tokenizer **char**, o vocab vem de `data/<nome_do_projeto>_char/vocab.txt` (gerado no Prepare a partir do `metadata.csv`). Com tokenizer **custom**, o vocab vem do arquivo indicado em **Tokenizer File**.
- **Conclusão:** Para o firstpixelptbr carregar e continuar treinando sem corromper, o **vocab usado no treino tem de ser idêntico** ao com que o firstpixelptbr foi treinado (mesmo arquivo ou mesma ordem e conjunto de caracteres).

---

## 2. O que você precisa ter

| Item | Onde | Observação |
|------|------|------------|
| Checkpoint firstpixelptbr | `ckpts/firstpixelptbr/model_last.pt` | Já usado no seu fluxo |
| Vocab do firstpixelptbr | `ckpts/firstpixelptbr/vocab.txt` | **Necessário** para alinhar o projeto; se não existir, obtenha de quem gerou o firstpixelptbr |

Sem o `vocab.txt` do firstpixelptbr não dá para garantir que o vocab do seu projeto está igual ao do base; treinar com vocab diferente é a causa mais provável de áudio em ruído.

---

## 3. O que fazer no projeto para funcionar corretamente

### Antes de treinar

1. **Obter/criar `ckpts/firstpixelptbr/vocab.txt`**  
   - Conteúdo: uma linha por caractere; primeira linha = espaço (índice 0).

2. **Preparar o dataset do projeto**  
   - Criar projeto (ex.: `meu_projeto_char`), colocar `wavs/` e `metadata.csv`, rodar **Prepare** com tokenizer **char**.

3. **Igualar o vocab do projeto ao do firstpixelptbr**  
   - Após o Prepare, **substituir** `data/meu_projeto_char/vocab.txt` por uma **cópia** de `ckpts/firstpixelptbr/vocab.txt`.  
   - Verificar que **todos os caracteres** que aparecem no `metadata.csv` existem nesse vocab; se faltar algum, ajustar o metadata ou usar “Vocab extend” com base no firstpixelptbr (se disponível na sua versão).

### No treino (Gradio)

- **Path to the Pretrained Checkpoint:** `ckpts/firstpixelptbr/model_last.pt` (ou path absoluto, ex. no Docker: `/workspace/F5-TTS/ckpts/firstpixelptbr/model_last.pt`).
- **Model:** F5TTS_Base.
- **Tokenizer:** char (recomendado após substituir o vocab do projeto pelo do firstpixelptbr).
- **Tokenizer File:** vazio (o vocab será lido de `data/<projeto>_char/vocab.txt`, que você já igualou ao do firstpixelptbr).

### Na inferência

- **Model (Custom):** `ckpts/<projeto>/model_XXX.pt` (ex.: `ckpts/brpb01_g2af01/model_600.pt`).
- **Vocab:** `data/<projeto>/vocab.txt` (o mesmo que você usou no treino = cópia do firstpixelptbr).
- **Config:** F5-TTS Base (dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4).

---

## 4. Checklist rápido

- [ ] Existe `ckpts/firstpixelptbr/vocab.txt` (ou você tem esse arquivo e vai colocá-lo aí).
- [ ] Após o Prepare, o `vocab.txt` do projeto foi **substituído** pelo do firstpixelptbr.
- [ ] Todos os caracteres do `metadata.csv` estão no `vocab.txt` do firstpixelptbr.
- [ ] No treino: pretrain = firstpixelptbr, tokenizer = char, tokenizer file vazio.
- [ ] Na inferência: mesmo vocab do projeto e config Base.

---

## 5. Referência no repositório

- **Guia passo a passo:** [FIRSTPIXELPTBR.md](FIRSTPIXELPTBR.md)  
- **Checkpoints e firstpixelptbr:** [ckpts/README.md](ckpts/README.md)

Com isso o projeto fica configurado para treinar baseado no firstpixelptbr e reduzir o risco de áudio quebrado por causa de vocab ou config errados.
