#!/usr/bin/env python3
"""
Gera 10 WAVs com ckpts/pesos_aleatorios/model_last.pt:
  - 5 com texto de geração aleatório (referência = clip aleatório do treino)
  - 5 com texto igual ao do clip de treino (ref e gen alinhados ao mesmo sample)

Escreve infer_out_pesos_aleatorios/manifest.csv (numerado, UTF-8) e copia cada WAV de
referência do treino para ref_original_XX_segment_....wav na mesma pasta (relatório).

  python tools/infer_pesos_aleatorios_batch.py
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from datasets import Dataset as HFDataset
from importlib.resources import files
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
)
from f5_tts.model import DiT


def _text_row_to_str(text) -> str:
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        return "".join(text).strip()
    return str(text).strip()


def _row_ref_audio(ds: HFDataset, data_root: Path, idx: int) -> tuple[str | None, str, Path | None]:
    """
    Resolve áudio de referência para a linha idx.
    Devolve (path ou None, texto, path temporário a apagar se áudio veio de array embutido).
    """
    row = ds[idx]
    ref_text = _text_row_to_str(row["text"])
    tmp_to_remove: Path | None = None
    wavs = data_root / "wavs"

    if "audio" in row and row["audio"] is not None and isinstance(row["audio"], dict):
        audio_arr = row["audio"]["array"]
        sr = int(row["audio"]["sampling_rate"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            p = Path(tmp.name)
        sf.write(str(p), audio_arr, sr)
        tmp_to_remove = p
        return str(p), ref_text, tmp_to_remove

    ap = row.get("audio_path")
    if not ap:
        return None, ref_text, None
    for cand in (Path(ap), wavs / Path(str(ap)).name, data_root / Path(str(ap)).name):
        try:
            if cand.is_file():
                return str(cand.resolve()), ref_text, None
        except OSError:
            continue
    return None, ref_text, None


def _collect_valid_indices(ds: HFDataset, data_root: Path) -> list[int]:
    valid: list[int] = []
    for i in range(len(ds)):
        path, _, _ = _row_ref_audio(ds, data_root, i)
        if path is not None and os.path.isfile(path):
            valid.append(i)
    return valid


# Cinco frases independentes do dataset (teste de generalização)
_RANDOM_GEN_TEXTS = [
    "Hoje o tempo está estável e as nuvens passam devagar sobre a cidade.",
    "Precisamos combinar horário, local e duração antes de confirmar a reunião.",
    "O mapa antigo guardava segredos entre dobras quase invisíveis ao olhar.",
    "Três ideias simples podem resolver metade dos problemas do projeto.",
    "Quando a luz falha, ainda resta o som da voz para guiar o próximo passo.",
]


def main() -> None:
    ckpt = _REPO / "ckpts" / "pesos_aleatorios" / "model_last.pt"
    vocab = _REPO / "data" / "pesos_aleatorios_char" / "vocab.txt"
    raw_arrow = _REPO / "data" / "pesos_aleatorios_char" / "raw.arrow"
    out_dir = _REPO / "infer_out_pesos_aleatorios"

    for p in (ckpt, vocab, raw_arrow):
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo necessário não encontrado: {p}")

    seed = int(os.environ.get("INFER_SEED", "42"))
    random.seed(seed)

    data_root = _REPO / "data" / "pesos_aleatorios_char"
    ds = HFDataset.from_file(str(raw_arrow))

    print("[infer] A recolher índices com ficheiro de áudio válido...", flush=True)
    valid = _collect_valid_indices(ds, data_root)
    if len(valid) < 10:
        raise RuntimeError(
            f"São necessários pelo menos 10 samples com WAV acessível; encontrei {len(valid)}."
        )

    chosen = random.sample(valid, 10)
    # Ordem: 5 primeiros -> texto aleatório; 5 últimos -> texto do treino
    idx_random = chosen[:5]
    idx_training = chosen[5:]

    rows_manifest: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    vocoder_local = _REPO / "checkpoints" / "vocos-mel-24khz"
    is_local = vocoder_local.is_dir()
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=is_local,
        local_path=str(vocoder_local) if is_local else "../checkpoints/vocos-mel-24khz",
    )

    model_cfg_path = files("f5_tts").joinpath("configs/F5TTS_Base_train.yaml")
    model_cfg = OmegaConf.load(str(model_cfg_path)).model.arch
    ema_model = load_model(
        DiT,
        model_cfg,
        str(ckpt),
        mel_spec_type=mel_spec_type,
        vocab_file=str(vocab),
    )

    def run_one(
        num: int,
        row_idx: int,
        gen_text: str | None,
        category: str,
    ) -> None:
        ref_path, ref_text, tmp_ref = _row_ref_audio(ds, data_root, row_idx)
        if not ref_path or not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Sem áudio para row {row_idx}")
        if gen_text is None:
            gen_text = ref_text
        ref_basename = Path(ref_path).name
        try:
            ref_audio, ref_text_ready = preprocess_ref_audio_text(ref_path, ref_text)
            print(f"\n--- [{num}/10] {category} ---\nref: {ref_basename}\nref_text: {ref_text[:120]}{'...' if len(ref_text) > 120 else ''}\ngen: {gen_text[:120]}{'...' if len(gen_text) > 120 else ''}\n", flush=True)
            audio_segment, final_sr, _ = infer_process(
                ref_audio,
                ref_text_ready,
                gen_text,
                ema_model,
                vocoder,
                mel_spec_type=mel_spec_type,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
            )
            out_wav = out_dir / f"pesos_aleatorios_{num:02d}.wav"
            sf.write(str(out_wav), audio_segment, final_sr)
            print(f"Salvo: {out_wav}", flush=True)
            ref_copy_name = f"ref_original_{num:02d}_{ref_basename}"
            ref_copy_path = out_dir / ref_copy_name
            shutil.copy2(ref_path, ref_copy_path)
            print(f"Cópia ref treino: {ref_copy_path}", flush=True)
            rows_manifest.append(
                {
                    "numero": str(num),
                    "ficheiro_wav": out_wav.name,
                    "indice_dataset": str(row_idx),
                    "ref_wav": ref_basename,
                    "ref_original_copia_relatorio": ref_copy_name,
                    "texto_referencia_treino": ref_text,
                    "texto_sintetizado": gen_text,
                    "tipo": category,
                }
            )
        finally:
            if tmp_ref is not None and tmp_ref.is_file():
                tmp_ref.unlink(missing_ok=True)

    # 1–5: texto aleatório
    for k, row_idx in enumerate(idx_random, start=1):
        run_one(k, row_idx, _RANDOM_GEN_TEXTS[k - 1], "texto_aleatorio")

    # 6–10: texto sintetizado = transcrição do treino (mesmo clip)
    for k, row_idx in enumerate(idx_training, start=6):
        run_one(k, row_idx, None, "texto_treino")

    manifest_path = out_dir / "manifest.csv"
    fieldnames = [
        "numero",
        "ficheiro_wav",
        "indice_dataset",
        "ref_wav",
        "ref_original_copia_relatorio",
        "texto_referencia_treino",
        "texto_sintetizado",
        "tipo",
    ]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows_manifest)

    print(f"\nManifest: {manifest_path}", flush=True)
    print(f"Concluído (seed={seed}). Saídas em: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
