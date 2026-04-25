import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from importlib.resources import files

import torch
from cached_path import cached_path
from datasets import Dataset as Dataset_
from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


def _configure_stdio_for_live_logs() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except (OSError, ValueError, AttributeError):
                pass


class _StreamTee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _dataset_stats(dataset_name: str, tokenizer: str) -> dict:
    data_suffix = "char" if tokenizer == "custom" else tokenizer
    rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{data_suffix}"))
    stats = {
        "data_path": rel_data_path,
        "total_read": 0,
        "total_used_estimate": 0,
        "total_discarded": 0,
        "discard_reasons": {
            "duration_out_of_range_(<0.3_or_>30s)": 0,
            "missing_audio_file": 0,
        },
    }
    raw_arrow = os.path.join(rel_data_path, "raw.arrow")
    if not os.path.isfile(raw_arrow):
        stats["note"] = f"raw.arrow not found at {raw_arrow}"
        return stats
    try:
        ds = Dataset_.from_file(raw_arrow)
        stats["total_read"] = len(ds)
        used = 0
        missing_audio = 0
        for row in ds:
            duration = float(row.get("duration", 0))
            if not (0.3 <= duration <= 30):
                stats["discard_reasons"]["duration_out_of_range_(<0.3_or_>30s)"] += 1
                continue
            audio_path = row.get("audio_path", "")
            if not audio_path or not os.path.isfile(audio_path):
                missing_audio += 1
                continue
            used += 1
        stats["discard_reasons"]["missing_audio_file"] = missing_audio
        stats["total_used_estimate"] = used
        stats["total_discarded"] = stats["total_read"] - used
    except Exception as e:  # noqa: BLE001
        stats["note"] = f"failed to compute detailed stats: {type(e).__name__}: {e}"
    return stats


def _artifact_checklist(checkpoint_path: str) -> dict:
    exp_dir = os.path.join(checkpoint_path, "experiment_report")
    samples_dir = os.path.join(checkpoint_path, "samples")
    graphics_dir = os.path.join(checkpoint_path, "graphics")
    has_samples = os.path.isdir(samples_dir) and any(p.endswith("_gen.wav") for p in os.listdir(samples_dir))
    return {
        "checkpoint_or_adapter_final": os.path.isfile(os.path.join(checkpoint_path, "model_last.pt")),
        "tempo_medio_por_epoca": os.path.isfile(os.path.join(exp_dir, "statistics_summary.json")),
        "tempo_medio_por_step": os.path.isfile(os.path.join(exp_dir, "statistics_summary.json")),
        "grafico_perda_por_epoca": os.path.isfile(os.path.join(graphics_dir, "loss_by_epoch.png"))
        or os.path.isfile(os.path.join(exp_dir, "loss_by_epoch.png")),
        "grafico_perda_por_step": os.path.isfile(os.path.join(graphics_dir, "loss.png"))
        or os.path.isfile(os.path.join(exp_dir, "loss.png")),
        "inferencia_por_checkpoint_salvo": has_samples,
    }


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    # batch_size_per_gpu = 1000 settting for gpu 8GB
    # batch_size_per_gpu = 1600 settting for gpu 12GB
    # batch_size_per_gpu = 2000 settting for gpu 16GB
    # batch_size_per_gpu = 3200 settting for gpu 24GB

    # num_warmup_updates = 300 for 5000 sample about 10 hours

    # change save_per_updates , last_per_steps change this value what you need  ,

    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name", type=str, default="F5TTS_Base", choices=["F5TTS_Base", "E2TTS_Base"], help="Experiment name"
    )
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size per GPU")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=300, help="Warmup steps")
    parser.add_argument("--save_per_updates", type=int, default=10000, help="Save checkpoint every X steps")
    parser.add_argument("--save_every_epochs", type=int, default=2, help="Save checkpoint at end of every N epochs (0 = use save_per_updates only)")
    parser.add_argument("--last_per_steps", type=int, default=50000, help="Save last checkpoint every X steps")
    parser.add_argument("--finetune", action="store_true", help="Use Finetune")
    parser.add_argument("--pretrain", type=str, default=None, help="the path to the checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=True,
        help="Log inferenced samples per ckpt save steps (default: True)",
    )
    parser.add_argument(
        "--no_log_samples",
        action="store_true",
        help="Disable log samples (override --log_samples)",
    )
    parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )
    parser.add_argument(
        "--experiment_report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CSV/JSON reports under ckpts/<dataset>/experiment_report.",
    )
    parser.add_argument(
        "--experiment_log_every_n",
        type=int,
        default=1,
        help="Write timing/loss entries every N steps.",
    )
    parser.add_argument(
        "--log_samples_every_n_epochs",
        type=int,
        default=0,
        help="Generate sample inference every N epochs (0=disabled).",
    )
    parser.add_argument(
        "--checkpoint_max_keep",
        type=int,
        default=10,
        help="Max numbered checkpoints to keep (0=no limit).",
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    _configure_stdio_for_live_logs()
    args = parse_args()
    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))
    os.makedirs(checkpoint_path, exist_ok=True)
    train_log_path = os.path.join(checkpoint_path, "train.log")
    train_log = open(train_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _StreamTee(sys.stdout, train_log)
    sys.stderr = _StreamTee(sys.stderr, train_log)

    run_name = f"{args.exp_name}_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("[finetune_cli] Starting finetune...", flush=True)
    print(f"[finetune_cli] train.log: {train_log_path}", flush=True)
    print(f"[finetune_cli] run_name: {run_name}", flush=True)
    print(f"[finetune_cli] base_model: {args.exp_name}", flush=True)
    print(
        f"[finetune_cli] base_checkpoint: {args.pretrain if args.pretrain else 'HF default (if --finetune) / random init'}",
        flush=True,
    )
    print(
        "[finetune_cli] hyperparams: "
        f"epochs={args.epochs}, learning_rate={args.learning_rate}, batch_size_per_gpu={args.batch_size_per_gpu}, "
        f"grad_accumulation_steps={args.grad_accumulation_steps}, max_samples={args.max_samples}, "
        f"batch_size_type={args.batch_size_type}, max_seq_len=NA",
        flush=True,
    )
    print(
        "[finetune_cli] tts_config: "
        f"vocoder={mel_spec_type}, sample_rate={target_sample_rate}, snac_repo_device=NA, text_loss_mask=NA",
        flush=True,
    )
    ds_stats = _dataset_stats(args.dataset_name, args.tokenizer)
    print("[finetune_cli] dataset_stats: " + json.dumps(ds_stats, ensure_ascii=False), flush=True)

    # Model parameters based on experiment name
    if args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain
    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    if args.finetune:
        # Normalize pretrain path: "workspace/..." -> "/workspace/..." when path does not exist
        if ckpt_path and not os.path.isfile(ckpt_path) and ckpt_path.startswith("workspace"):
            ckpt_path_abs = os.path.normpath("/" + ckpt_path)
            if os.path.isfile(ckpt_path_abs):
                ckpt_path = ckpt_path_abs

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.join(checkpoint_path, os.path.basename(ckpt_path))
        if not os.path.isfile(file_checkpoint):
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Pretrain checkpoint not found: {ckpt_path}. Use absolute path (e.g. /workspace/F5-TTS/ckpts/...)."
                )
            print(f"[finetune_cli] Copying pretrain checkpoint (may take a minute)...", flush=True)
            # When starting a NEW project from pretrain, save with step=0 so the trainer runs all epochs
            # (otherwise the pretrain file may contain step=199800 from another run and "no steps left")
            try:
                ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
                if isinstance(ckpt, dict) and "step" in ckpt:
                    ckpt["step"] = 0
                    torch.save(ckpt, file_checkpoint)
                    print("[finetune_cli] Pretrain copied with step reset to 0 (new run will do all epochs).", flush=True)
                else:
                    shutil.copy2(ckpt_path, file_checkpoint)
                    print("[finetune_cli] Copy checkpoint for finetune done.", flush=True)
                del ckpt
            except Exception:
                shutil.copy2(ckpt_path, file_checkpoint)
                print("[finetune_cli] Copy checkpoint for finetune done.", flush=True)

    # Use the tokenizer and tokenizer_path provided in the command line arguments
    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    print("[finetune_cli] Loading tokenizer...", flush=True)
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    print(f"[finetune_cli] Vocab size: {vocab_size}, vocoder: {mel_spec_type}", flush=True)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    print("[finetune_cli] Building model...", flush=True)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    print("[finetune_cli] Model built. Creating Trainer...", flush=True)

    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples and not args.no_log_samples,
        last_per_steps=args.last_per_steps,
        save_every_epochs=args.save_every_epochs,
        bnb_optimizer=args.bnb_optimizer,
        experiment_report=args.experiment_report,
        experiment_log_every_n_steps=args.experiment_log_every_n,
        log_samples_every_n_epochs=args.log_samples_every_n_epochs,
        checkpoint_max_keep=args.checkpoint_max_keep,
    )

    print("[finetune_cli] Loading dataset (this may take a while)...", flush=True)
    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    print(f"[finetune_cli] Dataset loaded, {len(train_dataset)} samples. Starting training loop...", flush=True)

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )

    checklist = _artifact_checklist(checkpoint_path)
    print("[finetune_cli] artifact_checklist_final: " + json.dumps(checklist, ensure_ascii=False), flush=True)
    missing = [k for k, v in checklist.items() if not v]
    if missing:
        print("[finetune_cli] artifact_checklist_status: MISSING -> " + ", ".join(missing), flush=True)
    else:
        print("[finetune_cli] artifact_checklist_status: OK (all expected artifacts present)", flush=True)


if __name__ == "__main__":
    main()
