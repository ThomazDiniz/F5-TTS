# Treinar com base firstpixelptbr

Este guia garante que o finetune use corretamente o checkpoint **firstpixelptbr** como base, evitando mismatch de vocabulário e áudio quebrado/ruído.

## Por que o vocab importa

O firstpixelptbr foi treinado com um **vocab.txt** específico (caracteres em português e ordem fixa). A camada de embedding do modelo tem tamanho `vocab_size + 1` (inclui token de preenchimento). Se você treinar com outro vocab (outro tamanho ou outra ordem), ao carregar o checkpoint:

- Ou o carregamento falha (shapes diferentes),
- Ou os índices dos caracteres não batem e o modelo recebe embeddings errados → áudio vira ruído.

Por isso o **vocab do projeto de treino tem de ser idêntico** ao do firstpixelptbr.

## O que você precisa no repositório

Na pasta de checkpoints do firstpixelptbr:

| Arquivo | Descrição |
|--------|-----------|
| `ckpts/firstpixelptbr/model_last.pt` | Checkpoint do modelo (obrigatório) |
| `ckpts/firstpixelptbr/vocab.txt` | Vocabulário usado no treino do firstpixelptbr (obrigatório para treinar a partir dele) |

Formato do `vocab.txt`: uma linha por caractere; a **primeira linha deve ser espaço** (índice 0 = unknown/space).

## Passos para configurar o projeto e treinar

### 1. Ter o vocab do firstpixelptbr

- Se você tem o `vocab.txt` do firstpixelptbr, coloque em `ckpts/firstpixelptbr/vocab.txt`.
- Se não tiver, é preciso obtê-lo de quem gerou o checkpoint firstpixelptbr; sem ele não dá para garantir compatibilidade.

### 2. Criar o projeto de dados

- Crie o projeto (ex.: `meu_projeto_char`) e coloque em `data/meu_projeto_char/`:
  - `wavs/` com os áudios (24 kHz recomendado),
  - `metadata.csv` (uma linha por amostra: `nome_arquivo|texto`).
- Rode **Prepare** com tokenizer **char** na interface de finetune. Isso gera `raw.arrow`, `duration.json` e um `vocab.txt` inicial em `data/meu_projeto_char/vocab.txt` (a partir dos caracteres do `metadata.csv`).

### 3. Alinhar o vocab do projeto ao do firstpixelptbr

- Confira se **todos os caracteres** que aparecem no seu `metadata.csv` existem no `ckpts/firstpixelptbr/vocab.txt`.
- Se existirem:
  - **Substitua** `data/meu_projeto_char/vocab.txt` por uma **cópia** de `ckpts/firstpixelptbr/vocab.txt`.
  - Assim o projeto passa a usar exatamente o mesmo vocabulário (e ordem) do firstpixelptbr e o checkpoint carrega sem mismatch.
- Se o firstpixelptbr **não** tiver algum caractere que você usa (ex.: símbolo raro):
  - Ou você remove/ajusta esse caractere no `metadata.csv`,  
  - Ou usa a função **“Vocab extend”** da UI com o checkpoint **firstpixelptbr** como base (se a sua versão da UI suportar primeiro checkpoint customizado; hoje a “Vocab extend” usa o modelo Emilia por padrão). Caso contrário, o seguro é usar apenas caracteres já presentes no `vocab.txt` do firstpixelptbr.

### 4. Treino na interface (Gradio)

- **Path to the Pretrained Checkpoint:** aponte para o firstpixelptbr, por exemplo:
  - Docker: `/workspace/F5-TTS/ckpts/firstpixelptbr/model_last.pt`
  - Local: caminho absoluto para `ckpts/firstpixelptbr/model_last.pt`
- **Tokenizer:** char (ou custom, se você apontar **Tokenizer File** para `ckpts/firstpixelptbr/vocab.txt` e tiver dados em pasta compatível; o fluxo mais simples é char + substituir o `vocab.txt` do projeto como acima).
- **Tokenizer File:** vazio se você já substituiu `data/meu_projeto_char/vocab.txt` pelo do firstpixelptbr; caso contrário, pode usar o caminho para `ckpts/firstpixelptbr/vocab.txt` se a UI usar esse arquivo como tokenizer.
- Demais parâmetros (learning rate, batch, epochs, save_per_updates, last_per_steps, etc.) conforme a documentação do finetune.

### 5. Inferência depois do treino

- Use sempre o **mesmo vocab** do projeto (que deve ser o do firstpixelptbr): em “Custom”, **Vocab** = `data/meu_projeto_char/vocab.txt` (ou o path que a interface usar para esse arquivo).
- **Config do modelo:** F5-TTS Base (dim=1024, depth=22, heads=16, …), igual ao primeiro experimento.

## Resumo rápido

1. Tenha `ckpts/firstpixelptbr/model_last.pt` e `ckpts/firstpixelptbr/vocab.txt`.
2. Prepare o projeto com tokenizer char.
3. Substitua o `vocab.txt` do projeto pelo `vocab.txt` do firstpixelptbr (garantindo que todos os caracteres do seu metadata estejam nesse vocab).
4. Treine com pretrain = firstpixelptbr e tokenizer char (e tokenizer file vazio, se o vocab do projeto já for o do firstpixelptbr).
5. Na inferência, use o vocab do projeto e a config Base.

Assim o treino fica alinhado ao firstpixelptbr e a continuação tende a funcionar corretamente em vez de gerar ruído por mismatch de embedding.
