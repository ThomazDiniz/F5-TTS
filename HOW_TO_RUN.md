# How to Install and Run F5-TTS

## Requisitos

- **Windows** (este guia e os `.bat` são para Windows).
- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda instalado.
- **GPU NVIDIA** com drivers e CUDA 11.8 (recomendado para inferência/treino).

---

## Instalação

### Opção 1: Instalação automática (recomendado)

1. Abra o **Explorador de Arquivos** e vá até a pasta do projeto (ex.: `E:\git\F5-TTS-FirstPixelToTRAIN`).
2. Dê **duplo clique** em **`install.bat`**.
3. Aguarde o fim da instalação (criação do ambiente, PyTorch e dependências). Pode levar alguns minutos.
4. Quando aparecer "Instalação concluída", a instalação está pronta.

O script `install.bat`:

- Detecta Miniconda/Anaconda no seu usuário ou em `C:\ProgramData`.
- Remove o ambiente `f5-tts` antigo, se existir.
- Cria o ambiente `f5-tts` com Python 3.10 a partir de `environment.yml`.
- Instala PyTorch 2.4 com CUDA 11.8.
- Instala o projeto em modo editável (`pip install -e .`).

### Opção 2: Instalação manual

Abra o **Anaconda Prompt** (ou um terminal com `conda` no PATH) e rode:

```bash
cd E:\git\F5-TTS-FirstPixelToTRAIN

conda env create -f environment.yml
conda activate f5-tts

pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Se a GPU for outra versão de CUDA, use o índice correspondente (ex.: `cu121` para CUDA 12.1). Para **apenas CPU**:

```bash
pip install torch==2.4.0 torchaudio==2.4.0
pip install -e .
```

---

## Como rodar

### Pelo Explorador de Arquivos (duplo clique)

| Arquivo               | Uso                          |
|-----------------------|------------------------------|
| **run_project.bat**   | Interface principal (inferência TTS) |
| **run_inference.bat** | Interface de inferência (Gradio)     |
| **run_finetune.bat**  | Interface de finetune (Gradio)      |

Dê duplo clique no `.bat` desejado. Uma janela do navegador deve abrir com a interface (geralmente em `http://127.0.0.1:7860`).

Os `.bat` assumem Miniconda em `%USERPROFILE%\miniconda3`. Se o Conda estiver em outro lugar (ex.: Anaconda em `%USERPROFILE%\anaconda3`), edite a linha `call "..."\activate.bat` no `.bat` e ajuste o caminho.

### Pelo terminal

```bash
conda activate f5-tts

# Inferência (interface web)
f5-tts_infer-gradio

# Finetune (interface web)
f5-tts_finetune-gradio

# Inferência por linha de comando
f5-tts_infer-cli --model "F5-TTS" --ref_audio "ref.wav" --ref_text "Texto do áudio." --gen_text "Texto a sintetizar."
```

---

## Resumo dos arquivos

| Arquivo          | Descrição                                      |
|------------------|------------------------------------------------|
| **install.bat**  | Instalação completa (conda, env, PyTorch, projeto). |
| **run_project.bat**  | Abre a interface de inferência (TTS).          |
| **run_inference.bat**| Abre a interface de inferência (TTS).          |
| **run_finetune.bat** | Abre a interface de finetune.                 |
| **environment.yml**  | Definição do ambiente conda `f5-tts`.         |

---

## Problemas comuns

- **"Conda não encontrado"**  
  Instale o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) e rode `install.bat` de novo.

- **`.bat` não ativa o ambiente**  
  Edite o `.bat` e troque `%USERPROFILE%\miniconda3` pelo caminho onde o Conda está (ex.: `%USERPROFILE%\anaconda3`).

- **Erro de CUDA / GPU**  
  Confirme que os drivers da NVIDIA e a versão de CUDA batem com o PyTorch (ex.: CUDA 11.8 para `cu118`). Para usar só CPU, instale `torch` e `torchaudio` sem sufixo `+cu118`.

- **Porta 7860 em uso**  
  Feche outro app que use a mesma porta ou inicie o Gradio com outra:  
  `f5-tts_infer-gradio --port 7861`
