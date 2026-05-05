import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return a[:n], b[:n]


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = _align(a, b)
    diff = a - b
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    corr = float(np.dot(a, b) / denom)
    return {"mse": mse, "mae": mae, "corr": corr}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "ckpts" / "filtered" / "outputs"
    report_dir = repo_root / "ckpts" / "filtered" / "experiment_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "inferencias_filtered.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("inferencias_filtered.csv is empty")

    cols = [c for c in rows[0].keys() if c != "texto"]
    if "checkpoint_original" not in cols:
        raise ValueError("CSV must contain 'checkpoint_original' column for comparison baseline.")
    model_cols = [c for c in cols if c != "checkpoint_original"]

    per_row_metrics: list[dict[str, str | float]] = []
    agg = {m: {"mse": [], "mae": [], "corr": []} for m in model_cols}

    for idx, row in enumerate(rows, start=1):
        ref_file = out_dir / str(row["checkpoint_original"])
        ref_wav, ref_sr = _load_mono(ref_file)

        for model_name in model_cols:
            wav_file = out_dir / str(row[model_name])
            wav, sr = _load_mono(wav_file)
            if sr != ref_sr:
                # Keep simple: compare first common samples after naive resample-by-truncation isn't valid.
                # We skip if SR differs; current pipeline should keep all at 24k.
                continue
            m = _metrics(wav, ref_wav)
            agg[model_name]["mse"].append(m["mse"])
            agg[model_name]["mae"].append(m["mae"])
            agg[model_name]["corr"].append(m["corr"])
            per_row_metrics.append(
                {
                    "row": idx,
                    "texto": row["texto"],
                    "model": model_name,
                    "file_model": row[model_name],
                    "file_reference": row["checkpoint_original"],
                    "mse_vs_checkpoint_original": m["mse"],
                    "mae_vs_checkpoint_original": m["mae"],
                    "corr_vs_checkpoint_original": m["corr"],
                }
            )

    detail_csv = report_dir / "post_eval_filtered_vs_original_by_row.csv"
    with detail_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row",
                "texto",
                "model",
                "file_model",
                "file_reference",
                "mse_vs_checkpoint_original",
                "mae_vs_checkpoint_original",
                "corr_vs_checkpoint_original",
            ],
        )
        w.writeheader()
        w.writerows(per_row_metrics)

    summary_rows = []
    for model_name in model_cols:
        summary_rows.append(
            {
                "model": model_name,
                "mean_mse_vs_checkpoint_original": float(np.mean(agg[model_name]["mse"])) if agg[model_name]["mse"] else float("nan"),
                "mean_mae_vs_checkpoint_original": float(np.mean(agg[model_name]["mae"])) if agg[model_name]["mae"] else float("nan"),
                "mean_corr_vs_checkpoint_original": float(np.mean(agg[model_name]["corr"])) if agg[model_name]["corr"] else float("nan"),
                "n_rows": len(agg[model_name]["mse"]),
            }
        )

    summary_csv = report_dir / "post_eval_filtered_vs_original_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "mean_mse_vs_checkpoint_original",
                "mean_mae_vs_checkpoint_original",
                "mean_corr_vs_checkpoint_original",
                "n_rows",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    # Plots
    labels = [r["model"] for r in summary_rows]
    mean_mse = [r["mean_mse_vs_checkpoint_original"] for r in summary_rows]
    mean_corr = [r["mean_corr_vs_checkpoint_original"] for r in summary_rows]

    plt.figure(figsize=(9, 4))
    plt.bar(labels, mean_mse)
    plt.title("Mean MSE vs checkpoint_original")
    plt.ylabel("MSE (lower is closer)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "post_eval_mean_mse_vs_original.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(labels, mean_corr)
    plt.title("Mean Correlation vs checkpoint_original")
    plt.ylabel("Correlation (higher is closer)")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "post_eval_mean_corr_vs_original.png", dpi=150)
    plt.close()

    print(f"Saved: {detail_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {report_dir / 'post_eval_mean_mse_vs_original.png'}")
    print(f"Saved: {report_dir / 'post_eval_mean_corr_vs_original.png'}")


if __name__ == "__main__":
    main()
