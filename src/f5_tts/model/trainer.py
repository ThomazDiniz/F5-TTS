from __future__ import annotations

import gc
import os
import time

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists

# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        save_every_epochs=0,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        print(f"Using logger: {logger}")
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.save_every_epochs = save_every_epochs if save_every_epochs else 0
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                #optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                print(f"Saving last checkpoint at step {step} (may take a minute for large files)...", flush=True)
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}", flush=True)
            else:
                print(f"Saving checkpoint at step {step}...", flush=True)
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")
            del checkpoint
            gc.collect()
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith(".pt") for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        all_pt = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")]
        # Prefer model_last.pt, then latest model_XXXXX.pt by step
        def _step_from_name(x):
            n = "".join(filter(str.isdigit, x))
            return int(n) if n else 0

        if "model_last.pt" in all_pt:
            candidates = ["model_last.pt"] + sorted(
                [f for f in all_pt if f != "model_last.pt"],
                key=_step_from_name,
                reverse=True,
            )
        else:
            candidates = sorted(all_pt, key=_step_from_name, reverse=True)

        checkpoint = None
        latest_checkpoint = None
        for ckpt_name in candidates:
            path = f"{self.checkpoint_path}/{ckpt_name}"
            try:
                checkpoint = torch.load(path, weights_only=True, map_location="cpu")
                latest_checkpoint = ckpt_name
                if self.is_main:
                    print(f"Loaded checkpoint: {ckpt_name}")
                break
            except (RuntimeError, OSError) as e:
                if self.is_main:
                    print(f"Skip corrupted/incomplete checkpoint {ckpt_name}: {e}")
                continue
        if checkpoint is None:
            if self.is_main:
                print("No valid checkpoint found, starting from step 0.")
            return 0

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "step" in checkpoint:
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
#            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0

        del checkpoint
        gc.collect()
        return step

    def _generate_and_save_log_samples(self, global_step: int, batch: dict):
        """Generate and save ref/gen audio samples (only when log_samples=True and on main process)."""
        if not self.log_samples or not self.accelerator.is_local_main_process:
            return
        t0 = time.perf_counter()
        cfg_strength, nfe_step, sway_sampling_coef = self._log_samples_cfg
        vocoder = self._log_samples_vocoder
        log_samples_path = self._log_samples_path
        target_sample_rate = self._log_samples_target_sr
        ref_audio_len = batch["mel_lengths"][0]
        mel_spec = batch["mel"].permute(0, 2, 1)
        text_inputs = batch["text"]
        infer_text = [
            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
        ]
        print(f"[trainer] Log samples step {global_step}: preparing inputs (ref_frames={ref_audio_len.item()})...", flush=True)
        t1 = time.perf_counter()
        with torch.inference_mode():
            print(f"[trainer] Log samples step {global_step}: running diffusion sampling ({nfe_step} steps)...", flush=True)
            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                text=infer_text,
                duration=ref_audio_len * 2,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            t2 = time.perf_counter()
            print(f"[trainer] Log samples step {global_step}: diffusion done ({t2-t1:.1f}s), decoding with vocoder...", flush=True)
            generated = generated.to(torch.float32)
            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
            ref_mel_spec = batch["mel"][0].unsqueeze(0)
            if self.vocoder_name == "vocos":
                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                ref_audio = vocoder.decode(ref_mel_spec).cpu()
            elif self.vocoder_name == "bigvgan":
                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()
            t3 = time.perf_counter()
            print(f"[trainer] Log samples step {global_step}: vocoder done ({t3-t2:.1f}s), saving wav files...", flush=True)
        torchaudio.save(f"{log_samples_path}/step_{global_step}_gen.wav", gen_audio, target_sample_rate)
        torchaudio.save(f"{log_samples_path}/step_{global_step}_ref.wav", ref_audio, target_sample_rate)
        t4 = time.perf_counter()
        print(f"[trainer] Log samples saved for step {global_step}. Total: {t4-t0:.1f}s (diffusion: {t2-t1:.1f}s, vocoder: {t3-t2:.1f}s, save: {t4-t3:.1f}s)", flush=True)

    def _plot_loss_curve(self):
        """Plot training loss vs step and save to checkpoint_path/graphics/loss.png."""
        if not self._loss_history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[trainer] matplotlib not available, skipping loss plot.", flush=True)
            return
        steps = [h[0] for h in self._loss_history]
        losses = [h[1] for h in self._loss_history]
        graphics_dir = os.path.join(self.checkpoint_path, "graphics")
        os.makedirs(graphics_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, linewidth=0.8, color="#1f77b4")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss over time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(graphics_dir, "loss.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[trainer] Loss curve saved to {out_path}", flush=True)

    def train(self, train_dataset: Dataset, num_workers=4, resumable_with_seed: int = None):
        if self.accelerator.is_local_main_process:
            print("[trainer] train() started: preparing dataloader and scheduler...", flush=True)
        self._loss_history = []
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            self._log_samples_vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            self._log_samples_target_sr = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            self._log_samples_path = f"{self.checkpoint_path}/samples"
            self._log_samples_cfg = (cfg_strength, nfe_step, sway_sampling_coef)
            os.makedirs(self._log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )
        if self.accelerator.is_local_main_process:
            print("[trainer] Preparing dataloader and scheduler with accelerator...", flush=True)
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual steps = 1 gpu steps / gpus
        if self.accelerator.is_local_main_process:
            print("[trainer] Loading checkpoint (resume)...", flush=True)
        start_step = self.load_checkpoint()
        global_step = start_step
        if self.accelerator.is_local_main_process:
            print(f"[trainer] Starting from step {global_step}. Entering epoch loop.", flush=True)

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            total_steps_this_run = int(orig_epoch_step * self.epochs / self.grad_accumulation_steps)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            if skipped_epoch >= self.epochs or start_step >= total_steps_this_run:
                if self.accelerator.is_local_main_process:
                    print(
                        f"[trainer] Resume at step {start_step} is beyond current config: "
                        f"{orig_epoch_step} batches/epoch × {self.epochs} epochs = {total_steps_this_run} total steps. "
                        f"No training steps left. Saving and exiting. (Use a checkpoint from this dataset or increase epochs to train more.)",
                        flush=True,
                    )
                # Skip the epoch loop; go straight to final save
            else:
                skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0
            skipped_dataloader = None
            total_steps_this_run = int(len(train_dataloader) * self.epochs / self.grad_accumulation_steps)

        for epoch in range(skipped_epoch, self.epochs):
            if self.accelerator.is_local_main_process:
                print(f"[trainer] Epoch {epoch + 1}/{self.epochs} started.", flush=True)
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_step)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main and self.accelerator.sync_gradients:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self._loss_history.append((global_step, float(loss.item())))
                    self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_step)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_step)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                save_interval = self.save_per_updates * self.grad_accumulation_steps
                if save_interval > 0 and global_step % save_interval == 0:
                    self.save_checkpoint(global_step)
                    if self.log_samples and self.accelerator.is_local_main_process:
                        print(f"[trainer] Generating log samples at step {global_step} (may take a moment)...", flush=True)
                    self._generate_and_save_log_samples(global_step, batch)

                if self.last_per_steps > 0 and global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)
                    if self.log_samples and self.accelerator.is_local_main_process:
                        print(f"[trainer] Generating log samples at step {global_step} (may take a moment)...", flush=True)
                    self._generate_and_save_log_samples(global_step, batch)

            # Save at end of every N epochs (if set)
            if self.save_every_epochs > 0 and (epoch + 1) % self.save_every_epochs == 0:
                if self.accelerator.is_local_main_process:
                    print(f"[trainer] End of epoch {epoch + 1}: saving checkpoint (save every {self.save_every_epochs} epochs).", flush=True)
                self.save_checkpoint(global_step)
                self.save_checkpoint(global_step, last=True)
                if self.log_samples and self.accelerator.is_local_main_process:
                    print(f"[trainer] Generating log samples at step {global_step} (may take a moment)...", flush=True)
                self._generate_and_save_log_samples(global_step, batch)

        if self.accelerator.is_local_main_process:
            print("[trainer] Training loop finished. Saving final checkpoint...", flush=True)
        self.save_checkpoint(global_step, last=True)
        if self.accelerator.is_local_main_process:
            print("[trainer] Calling accelerator.end_training()...", flush=True)
        self.accelerator.end_training()
        if self.accelerator.is_local_main_process:
            if self._loss_history:
                self._plot_loss_curve()
            print("[trainer] Training done.", flush=True)
