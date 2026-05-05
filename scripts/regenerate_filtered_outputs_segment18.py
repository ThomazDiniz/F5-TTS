import csv
import hashlib
import io
import sys
from datetime import datetime
from pathlib import Path

from f5_tts.api import F5TTS


TEXTS = [
    "Hoje acordei cedo, preparei um café forte e organizei a mesa para estudar com calma.",
    "Enquanto o trem passava devagar, uma criança sorria e apontava para as nuvens alaranjadas.",
    "O pesquisador analisou os dados, escreveu um relatório objetivo e compartilhou as conclusões com a equipe.",
    "Estamos construindo uma rotina mais saudável, caminhando no bairro e cozinhando alimentos frescos todos os dias.",
    "A bibliotecária catalogou romances, dicionários e biografias, mantendo cada prateleira limpa e bem sinalizada.",
    "Se você estiver dirigindo com chuva intensa, reduza a velocidade, mantenha distância e evite manobras bruscas.",
    "No fim da tarde, os amigos conversavam rindo, lembrando histórias antigas e planejando uma viagem curta.",
    "A startup lançou um aplicativo simples, corrigiu bugs rapidamente e recebeu comentários positivos dos usuários.",
    "Quando o professor explicava o experimento, os alunos anotavam atentos e faziam perguntas bastante pertinentes.",
    "Ela estava finalizando o projeto, revisando detalhes técnicos e preparando a apresentação para segunda-feira.",
]
NUM_TEXTS = 5

# Áudio de referência na raiz do repo (pedido: PB_0991).
REF_AUDIO_NAME = "PB_0991.wav"


class Tee(io.TextIOBase):
    """Write to console and log file at the same time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "ckpts" / "filtered" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = repo_root / "ckpts" / "filtered"
    data_dir = repo_root / "data" / "filtered_char"
    vocab_file = data_dir / "vocab.txt"

    ref_audio = repo_root / REF_AUDIO_NAME
    # Vazio: o F5-TTS transcreve o áudio de referência em preprocess_ref_audio_text.
    ref_text = ""

    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    log_path = out_dir / f"inferencias_filtered_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8")
    tee = Tee(original_stdout, log_file)
    sys.stdout = tee
    sys.stderr = tee
    print(f"[RUN] log_file={log_path}")

    # model_last.pt omite-se: pesos idênticos a model_52926.pt (mesmo model_state_dict).
    checkpoints = [
        ("model_17642.pt", ckpt_dir / "model_17642.pt", str(vocab_file), False),
        ("model_35284.pt", ckpt_dir / "model_35284.pt", str(vocab_file), False),
        ("model_52926.pt", ckpt_dir / "model_52926.pt", str(vocab_file), False),
        (
            "checkpoint_original",
            repo_root / "ckpts" / "firstpixelptbr" / "model_last.safetensors",
            str(repo_root / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt"),
            True,
        ),
    ]

    try:
        tts_map: dict[str, F5TTS] = {}
        for col_name, ckpt_path, vocab, use_ema in checkpoints:
            print(f"[load] {col_name} | ckpt={ckpt_path} | exists={ckpt_path.exists()} | size={ckpt_path.stat().st_size if ckpt_path.exists() else 'N/A'}")
            if ckpt_path.exists():
                print(f"[load] {col_name} | ckpt_md5={md5_file(ckpt_path)}")
            tts_map[col_name] = F5TTS(
                model_type="F5-TTS",
                ckpt_file=str(ckpt_path),
                vocab_file=vocab,
                use_ema=use_ema,
            )

        rows: list[dict[str, str]] = []
        for i, text in enumerate(TEXTS[:NUM_TEXTS], start=1):
            row: dict[str, str] = {"texto": text}
            rows.append(row)
            phrase_md5: dict[str, str] = {}
            for col_name, *_ in checkpoints:
                out_wav = out_dir / f"{col_name}_frase_{i:02d}.wav"
                print(f"[infer] {col_name} | frase {i:02d} | out={out_wav}")
                tts_map[col_name].infer(
                    ref_file=str(ref_audio),
                    ref_text=ref_text,
                    gen_text=text,
                    file_wave=str(out_wav),
                    speed=1.0,
                    nfe_step=32,
                    remove_silence=False,
                    seed=1234 + i,
                )
                row[col_name] = out_wav.name
                if out_wav.exists():
                    phrase_md5[col_name] = md5_file(out_wav)
                    print(f"[infer] {col_name} | frase {i:02d} | wav_md5={phrase_md5[col_name]} | bytes={out_wav.stat().st_size}")
                else:
                    print(f"[infer][ERROR] output missing: {out_wav}")

            # Detect identical output within same phrase across different checkpoints.
            inv: dict[str, list[str]] = {}
            for model_name, md5 in phrase_md5.items():
                inv.setdefault(md5, []).append(model_name)
            for md5, models in inv.items():
                if len(models) > 1:
                    print(f"[infer][WARN] frase {i:02d} identical wav across models: {models} | md5={md5}")

        csv_path = out_dir / "inferencias_filtered.csv"
        fieldnames = ["texto"] + [col for col, *_ in checkpoints]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        slash_path = out_dir / "inferencias_filtered_slash.csv"
        header = "texto/" + "//".join([col for col, *_ in checkpoints])
        lines = [header]
        for row in rows:
            vals = [row[col] for col, *_ in checkpoints]
            lines.append(f"{row['texto']}/" + "//".join(vals))
        slash_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

        print(f"Done. reference={ref_audio} (ref_text vazio → STT automático no preprocess)")
        print(f"CSV: {csv_path}")
        print(f"CSV slash: {slash_path}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()
        print(f"[RUN] full log saved at {log_path}")


if __name__ == "__main__":
    main()
