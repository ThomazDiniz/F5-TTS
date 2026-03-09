#!/usr/bin/env python3
"""
Analisa um projeto de treino (ex.: brPB22_g1bF01_char) e reporta problemas
nos arquivos. Rodar na raiz do repo ou com PYTHONPATH incluindo o projeto.

Uso:
  python scripts/check_training_project.py brPB22_g1bF01_char
  python scripts/check_training_project.py brpb22_g1bf01_char
"""
import json
import os
import sys
from pathlib import Path

# project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def find_project_folder(name: str) -> Path | None:
    """Encontra a pasta do projeto em data/ (case-insensitive no Windows)."""
    name_clean = name.replace("_pinyin", "").replace("_char", "").strip().lower()
    if not DATA_DIR.is_dir():
        return None
    for folder in os.listdir(DATA_DIR):
        path = DATA_DIR / folder
        if not path.is_dir():
            continue
        if folder.lower() == name.lower() or folder.lower().replace("_pinyin", "").replace("_char", "") == name_clean:
            return path
    # exact match
    exact = DATA_DIR / name
    if exact.is_dir():
        return exact
    return None


def get_audio_duration(path: Path) -> float | None:
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        return wav.shape[1] / sr
    except Exception:
        return None


def run_checks(project_path: Path) -> list[str]:
    issues = []
    path_wavs = project_path / "wavs"
    file_metadata = project_path / "metadata.csv"
    file_raw = project_path / "raw.arrow"
    file_duration = project_path / "duration.json"
    file_vocab = project_path / "vocab.txt"

    # 1. metadata.csv
    if not file_metadata.is_file():
        issues.append(f"Arquivo não encontrado: {file_metadata}")
        return issues
    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        issues.append("metadata.csv está vazio.")
        return issues
    bad_lines = []
    for i, line in enumerate(lines):
        parts = line.split("|")
        if len(parts) != 2:
            bad_lines.append(f"  Linha {i+1}: esperado 'nome|texto', obtido {len(parts)} partes")
            continue
        name_audio, text = parts[0].strip(), parts[1].strip()
        if len(text) < 3:
            bad_lines.append(f"  Linha {i+1}: texto muito curto (< 3 caracteres)")
    if bad_lines:
        issues.append("metadata.csv – problemas:\n" + "\n".join(bad_lines[:20]))
        if len(bad_lines) > 20:
            issues.append(f"  ... e mais {len(bad_lines) - 20} linhas com problema.")

    # 2. wavs/
    if not path_wavs.is_dir():
        issues.append(f"Pasta não encontrada: {path_wavs}")
    else:
        wav_files = list(path_wavs.glob("*.wav")) + list(path_wavs.glob("*.mp3"))
        if not wav_files:
            issues.append(f"Nenhum áudio .wav ou .mp3 em {path_wavs}")
        else:
            # check referenced files from metadata
            missing = []
            for line in lines:
                parts = line.split("|")
                if len(parts) != 2:
                    continue
                name = parts[0].strip()
                if not name:
                    continue
                found = False
                for ext in ("wav", "mp3", "flac", "m4a"):
                    if (path_wavs / f"{name}.{ext}").exists() or (path_wavs / name).exists():
                        found = True
                        break
                if name.endswith((".wav", ".mp3")):
                    found = (path_wavs / name).exists()
                if not found:
                    missing.append(name)
            if missing:
                issues.append(f"Áudios referenciados no metadata mas ausentes em wavs/ ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # 3. raw.arrow (após Prepare)
    if not file_raw.is_file():
        issues.append(f"Arquivo raw.arrow não encontrado: {file_raw} (rode Prepare na UI de finetune)")
    else:
        try:
            from datasets import Dataset
            ds = Dataset.from_file(str(file_raw))
            if len(ds) == 0:
                issues.append("raw.arrow existe mas está vazio (0 amostras).")
        except ImportError:
            if file_raw.stat().st_size == 0:
                issues.append("raw.arrow existe mas está vazio (0 bytes).")
        except Exception as e:
            issues.append(f"Erro ao abrir raw.arrow: {e}")

    # 4. duration.json
    if not file_duration.is_file():
        issues.append(f"Arquivo duration.json não encontrado: {file_duration} (gerado pelo Prepare)")
    else:
        try:
            with open(file_duration, "r", encoding="utf-8") as f:
                d = json.load(f)
            dur_list = d.get("duration", [])
            if not dur_list:
                issues.append("duration.json existe mas 'duration' está vazio.")
            else:
                out_of_range = [d for d in dur_list if not (1 <= d <= 30)]
                if out_of_range:
                    issues.append(f"duration.json: {len(out_of_range)} durações fora do intervalo [1, 30] segundos (ex.: {out_of_range[:3]}).")
        except Exception as e:
            issues.append(f"Erro ao ler duration.json: {e}")

    # 5. vocab.txt
    if not file_vocab.is_file():
        issues.append(f"Arquivo vocab.txt não encontrado: {file_vocab} (gerado pelo Prepare ou use o do firstpixelptbr)")
    else:
        with open(file_vocab, "r", encoding="utf-8-sig") as f:
            vocab_lines = [ln.rstrip("\n") for ln in f]
        if not vocab_lines:
            issues.append("vocab.txt está vazio.")
        else:
            if vocab_lines[0] != " ":
                issues.append("vocab.txt: a primeira linha deve ser espaço (índice 0).")

    return issues


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/check_training_project.py <nome_do_projeto>")
        print("Ex.: python scripts/check_training_project.py brPB22_g1bF01_char")
        sys.exit(1)
    name = sys.argv[1]
    project_path = find_project_folder(name)
    if project_path is None:
        print(f"[ERRO] Projeto '{name}' não encontrado em data/.")
        print(f"Pasta esperada: data/{name} ou data/<nome_similar>")
        print(f"Diretório data: {DATA_DIR}")
        if DATA_DIR.is_dir():
            print("Pastas em data/:", [p for p in os.listdir(DATA_DIR) if (DATA_DIR / p).is_dir()])
        sys.exit(2)
    print(f"Projeto encontrado: {project_path}")
    issues = run_checks(project_path)
    if not issues:
        print("Nenhum problema encontrado nos arquivos de treino.")
        return
    print("\nProblemas encontrados:")
    for i, msg in enumerate(issues, 1):
        print(f"\n{i}. {msg}")


if __name__ == "__main__":
    main()
