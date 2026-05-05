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
    out_dir = repo_root / "ckpts" / "filtered" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / "data" / "filtered_char"
    ref_audio, ref_text = load_first_reference(data_dir)

    print("[load] checkpoint_original (F5-TTS base)")
    tts_original = F5TTS(
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        use_ema=True,
    )

    generated_original = []
    for i, text in enumerate(TEXTS, start=1):
        out_wav = out_dir / f"checkpoint_original_frase_{i:02d}.wav"
        print(f"[infer] checkpoint_original | frase {i:02d}")
        tts_original.infer(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(out_wav),
            speed=1.0,
            nfe_step=32,
            remove_silence=False,
            seed=1234 + i,
        )
        generated_original.append(out_wav.name)

    csv_path = out_dir / "inferencias_filtered.csv"
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    for idx, row in enumerate(rows):
        row["checkpoint_original"] = generated_original[idx]

    fieldnames = list(rows[0].keys())
    if "checkpoint_original" not in fieldnames:
        fieldnames.append("checkpoint_original")
    else:
        fieldnames = [c for c in fieldnames if c != "checkpoint_original"] + ["checkpoint_original"]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    slash_path = out_dir / "inferencias_filtered_slash.csv"
    cols = [c for c in fieldnames if c != "texto"]
    slash_lines = ["texto/" + "//".join(cols)]
    for row in rows:
        slash_lines.append(row["texto"] + "/" + "//".join(row[c] for c in cols))
    slash_path.write_text("\n".join(slash_lines) + "\n", encoding="utf-8-sig")

    print(f"Done. Added checkpoint_original files in: {out_dir}")
    print(f"Updated CSV: {csv_path}")
    print(f"Updated slash CSV: {slash_path}")


if __name__ == "__main__":
    main()
