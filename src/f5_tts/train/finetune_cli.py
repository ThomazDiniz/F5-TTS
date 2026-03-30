import argparse
import os
import shutil

import torch
from cached_path import cached_path
from f5_tts.model import CFM, UNetT, DiT, Trainer
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset
from importlib.resources import files


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


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

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()
    print("[finetune_cli] Starting finetune...", flush=True)

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))

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
    )

    print("[finetune_cli] Loading dataset (this may take a while)...", flush=True)
    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    print(f"[finetune_cli] Dataset loaded, {len(train_dataset)} samples. Starting training loop...", flush=True)

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
