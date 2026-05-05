import csv
from pathlib import Path

from f5_tts.api import F5TTS


TEXTS = [
    "Hoje acordei cedo, preparei um cafe forte e organizei a mesa para estudar com calma.",
    "Enquanto o trem passava devagar, uma crianca sorria e apontava para as nuvens alaranjadas.",
    "O pesquisador analisou os dados, escreveu um relatorio objetivo e compartilhou as conclusoes com a equipe.",
    "Estamos construindo uma rotina mais saudavel, caminhando no bairro e cozinhando alimentos frescos todos os dias.",
    "A bibliotecaria catalogou romances, dicionarios e biografias, mantendo cada prateleira limpa e bem sinalizada.",
    "Se voce estiver dirigindo com chuva intensa, reduza a velocidade, mantenha distancia e evite manobras bruscas.",
    "No fim da tarde, os amigos conversavam rindo, lembrando historias antigas e planejando uma viagem curta.",
    "A startup lancou um aplicativo simples, corrigiu bugs rapidamente e recebeu comentarios positivos dos usuarios.",
    "Quando o professor explicava o experimento, os alunos anotavam atentos e faziam perguntas bastante pertinentes.",
    "Ela estava finalizando o projeto, revisando detalhes tecnicos e preparando a apresentacao para segunda-feira.",
]


def load_first_reference(project_data_dir: Path) -> tuple[Path, str]:
    metadata = project_data_dir / "metadata.csv"
    wavs_dir = project_data_dir / "wavs"
    first_line = metadata.read_text(encoding="utf-8-sig").splitlines()[0]
    seg_name, ref_text = first_line.split("|", 1)
    ref_audio = wavs_dir / f"{seg_name}.wav"
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    return ref_audio, ref_text.strip()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_dir = repo_root / "ckpts" / "filtered"
    data_dir = repo_root / "data" / "filtered_char"
    vocab_file = data_dir / "vocab.txt"
    out_dir = ckpt_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # model_last.pt omite-se: duplicado de model_52926.pt (mesmos pesos).
    ckpt_files = sorted(
        p for p in ckpt_dir.glob("model_*.pt")
        if p.name in {"model_17642.pt", "model_35284.pt", "model_52926.pt"}
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {ckpt_dir}")

    ref_audio, ref_text = load_first_reference(data_dir)
    print(f"Reference audio: {ref_audio}")
    print(f"Reference text: {ref_text}")

    tts_by_ckpt: dict[str, F5TTS] = {}
    for ckpt in ckpt_files:
        print(f"[load] {ckpt.name}")
        tts_by_ckpt[ckpt.name] = F5TTS(
            model_type="F5-TTS",
            ckpt_file=str(ckpt),
            vocab_file=str(vocab_file),
            use_ema=False,
        )

    rows: list[dict[str, str]] = []
    for i, text in enumerate(TEXTS, start=1):
        row: dict[str, str] = {"texto": text}
        rows.append(row)

        for ckpt in ckpt_files:
            model_name = ckpt.stem
            out_wav = out_dir / f"{model_name}_frase_{i:02d}.wav"
            print(f"[infer] {model_name} | frase {i:02d}")
            tts = tts_by_ckpt[ckpt.name]
            tts.infer(
                ref_file=str(ref_audio),
                ref_text=ref_text,
                gen_text=text,
                file_wave=str(out_wav),
                speed=1.0,
                nfe_step=32,
                remove_silence=False,
                seed=1234 + i,
            )
            row[ckpt.name] = out_wav.name

    csv_path = out_dir / "inferencias_filtered.csv"
    fieldnames = ["texto"] + [p.name for p in ckpt_files]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. CSV: {csv_path}")
    print(f"Audios in: {out_dir}")


if __name__ == "__main__":
    main()
