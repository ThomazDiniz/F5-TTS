import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _read_loss_csv(path: Path):
    steps = []
    losses = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["global_step"]))
                losses.append(float(row["loss"]))
            except (TypeError, ValueError, KeyError):
                continue
    return steps, losses


def _read_eval_summary(path: Path):
    points = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "")
            m = re.search(r"model_(\d+)\.pt", model)
            if not m:
                # ignore non-step checkpoints (ex.: checkpoint_original)
                continue
            step = int(m.group(1))
            points.append(
                {
                    "model": model,
                    "step": step,
                    "mean_corr": float(row["mean_corr_vs_checkpoint_original"]),
                    "mean_mse": float(row["mean_mse_vs_checkpoint_original"]),
                }
            )
    points.sort(key=lambda x: x["step"])
    return points


def main():
    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "ckpts" / "filtered" / "experiment_report"
    loss_csv = report_dir / "loss_by_step.csv"
    eval_csv = report_dir / "post_eval_filtered_vs_original_summary.csv"

    if not loss_csv.exists():
        raise FileNotFoundError(f"Missing {loss_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing {eval_csv}")

    steps, losses = _read_loss_csv(loss_csv)
    eval_points = _read_eval_summary(eval_csv)
    if not steps:
        raise ValueError("No loss points found in loss_by_step.csv")
    if not eval_points:
        raise ValueError("No eval points with model_<step>.pt found in eval summary.")

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(steps, losses, color="#1f77b4", linewidth=1.0, alpha=0.85, label="Train loss")
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("Train loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    e_steps = [p["step"] for p in eval_points]
    e_corr = [p["mean_corr"] for p in eval_points]
    ax2.plot(
        e_steps,
        e_corr,
        color="#d62728",
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="Post-eval corr vs original",
    )
    ax2.set_ylabel("Eval corr", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0.0, 1.0)

    for p in eval_points:
        ax2.annotate(
            p["model"],
            (p["step"], p["mean_corr"]),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=8,
            color="#d62728",
        )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.title("Filtered training loss + post-eval correlation")
    plt.tight_layout()

    out_png = report_dir / "loss_with_eval_corr_overlay.png"
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
