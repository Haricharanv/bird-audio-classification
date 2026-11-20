"""Complete PyTorch Trainer for Bird Audio Classification with mixed precision, warmup, and logging."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score
from torch.utils.data import DataLoader

from src.data.augment import BatchMixupCutmix
from src.utils.io import load_checkpoint, save_checkpoint
from src.utils.logging import format_epoch_table, get_logger

logger = get_logger("bird_classification.trainer")


class Trainer:
    """Production PyTorch Trainer managing the training lifecycle, evaluation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scheduler_step_mode: str,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Any,
        device: torch.device,
        species_to_idx: Dict[str, int],
        run_dir: Union[str, Path],
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_mode = scheduler_step_mode
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.species_to_idx = species_to_idx
        self.num_classes = len(species_to_idx)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = int(getattr(cfg.train, "epochs", 25))
        self.grad_accum_steps = max(1, int(getattr(cfg.train, "gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(getattr(cfg.train, "max_grad_norm", 1.0))
        self.freeze_epochs = int(getattr(cfg.model, "freeze_epochs", 1))
        self.early_stopping_patience = int(getattr(cfg.train, "early_stopping_patience", 7))
        self.monitor_metric = str(getattr(cfg.train, "monitor_metric", "val_macro_f1"))
        self.monitor_mode = str(getattr(cfg.train, "monitor_mode", "max"))
        self.is_multilabel = str(getattr(cfg, "task", "multiclass")).lower() == "multilabel"

        # Mixed precision setup
        mp_setting = str(getattr(cfg.system, "mixed_precision", "fp16")).lower()
        self.use_amp = mp_setting in ("fp16", "bf16") and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if mp_setting == "bf16" else torch.float16
        self.amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and mp_setting == "fp16"))

        # Batch Augmentations (Mixup / CutMix)
        batch_aug_cfg = cfg.augment.batch if hasattr(cfg, "augment") else {}
        self.mixup_cutmix = BatchMixupCutmix(
            mixup_alpha=float(getattr(batch_aug_cfg, "mixup_alpha", 0.4)),
            mixup_prob=float(getattr(batch_aug_cfg, "mixup_prob", 0.5)),
            cutmix_alpha=float(getattr(batch_aug_cfg, "cutmix_alpha", 0.4)),
            cutmix_prob=float(getattr(batch_aug_cfg, "cutmix_prob", 0.0)),
        )

        # Logging destinations
        self.csv_log_path = self.run_dir / "metrics.csv"
        self._init_csv_log()

        self.writer = None
        if bool(getattr(cfg.logging, "tensorboard", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")

        self.wandb_run = None
        if bool(getattr(cfg.logging, "wandb", False)):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=str(getattr(cfg.logging, "wandb_project", "birdclef-2025")),
                    name=str(getattr(cfg, "experiment_name", "run")),
                    config=cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")

        # State tracking
        self.best_metric = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_epoch = 1

        # Check for checkpoint resume
        resume_cp = getattr(cfg.train, "resume_checkpoint", None)
        if resume_cp and Path(resume_cp).is_file():
            self._resume(resume_cp)

    def _init_csv_log(self) -> None:
        """Create metrics CSV header."""
        headers = [
            "epoch", "train_loss", "val_loss", "val_macro_f1",
            "val_top1_acc", "val_top3_acc", "val_roc_auc", "lr", "epoch_time_sec"
        ]
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _resume(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training state from checkpoint."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        ckpt = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("metric_val", self.best_metric)
        logger.info(f"Resumed from epoch {self.start_epoch} (best metric: {self.best_metric:.4f}).")

    def train_epoch(self, epoch: int) -> float:
        """Execute one complete training epoch."""
        self.model.train()

        # Handle backbone freezing / unfreezing schedule
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "freeze_backbone") and hasattr(raw_model, "unfreeze_backbone"):
            if epoch <= self.freeze_epochs:
                raw_model.freeze_backbone()
                logger.info(f"[Epoch {epoch}] Backbone frozen for warmup.")
            else:
                raw_model.unfreeze_backbone()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            waveforms = batch["waveform"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            # Apply batch-level Mixup / CutMix
            if not self.is_multilabel:
                waveforms, targets = self.mixup_cutmix(waveforms, targets, num_classes=self.num_classes)

            # Mixed precision forward pass
            with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(waveforms)
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Optimizer step on accumulation boundary
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None and self.scheduler_step_mode == "step":
                    self.scheduler.step()

            total_loss += loss.item() * self.grad_accum_steps

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate model on the validation split."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for batch in self.val_loader:
            waveforms = batch["waveform"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(waveforms)
                loss = self.criterion(logits, targets)

            total_loss += loss.item()

            if self.is_multilabel:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        preds_arr = np.concatenate(all_preds, axis=0)      # (N, C)
        targets_arr = np.concatenate(all_targets, axis=0)  # (N,) or (N, C)

        # Metric computation
        metrics: Dict[str, float] = {"val_loss": val_loss}

        if self.is_multilabel:
            binary_preds = (preds_arr >= 0.5).astype(int)
            macro_f1 = float(f1_score(targets_arr, binary_preds, average="macro", zero_division=0))
            metrics["val_macro_f1"] = macro_f1
            metrics["val_top1_acc"] = float(np.mean(binary_preds == targets_arr))
            metrics["val_top3_acc"] = metrics["val_top1_acc"]
            metrics["val_roc_auc"] = 0.0
        else:
            pred_classes = np.argmax(preds_arr, axis=1)
            macro_f1 = float(f1_score(targets_arr, pred_classes, average="macro", zero_division=0))
            top1_acc = float(np.mean(pred_classes == targets_arr))

            # Top-3 Accuracy
            top3_k = min(3, self.num_classes)
            top3_acc = float(top_k_accuracy_score(targets_arr, preds_arr, k=top3_k, labels=list(range(self.num_classes))))

            # BirdCLEF-style macro ROC-AUC: average over classes with at least one positive in val
            roc_auc_scores = []
            for c in range(self.num_classes):
                y_true_c = (targets_arr == c).astype(int)
                if np.sum(y_true_c) > 0 and np.sum(y_true_c) < len(y_true_c):
                    try:
                        score = roc_auc_score(y_true_c, preds_arr[:, c])
                        roc_auc_scores.append(score)
                    except ValueError:
                        pass

            macro_roc_auc = float(np.mean(roc_auc_scores)) if roc_auc_scores else 0.0

            metrics["val_macro_f1"] = macro_f1
            metrics["val_top1_acc"] = top1_acc
            metrics["val_top3_acc"] = top3_acc
            metrics["val_roc_auc"] = macro_roc_auc

        return metrics

    def fit(self) -> Dict[str, Any]:
        """Execute full training and validation loop with early stopping and logging."""
        logger.info(
            f"Starting training on device '{self.device}' ({self.epochs} epochs, "
            f"batch_size={self.train_loader.batch_size}, num_classes={self.num_classes})."
        )

        headers = ["Epoch", "Train Loss", "Val Loss", "Macro-F1", "Top-1 Acc", "Top-3 Acc", "ROC-AUC", "LR", "Time (s)"]
        history_rows: List[List[Any]] = []

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            epoch_time = time.time() - t0

            # Step scheduler if epoch-based
            if self.scheduler is not None and self.scheduler_step_mode == "epoch":
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            val_f1 = val_metrics["val_macro_f1"]
            val_top1 = val_metrics["val_top1_acc"]
            val_top3 = val_metrics["val_top3_acc"]
            val_auc = val_metrics["val_roc_auc"]
            val_loss = val_metrics["val_loss"]

            # Save to CSV log
            with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, train_loss, val_loss, val_f1, val_top1, val_top3, val_auc, current_lr, epoch_time
                ])

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Metrics/Macro_F1", val_f1, epoch)
                self.writer.add_scalar("Metrics/Top1_Acc", val_top1, epoch)
                self.writer.add_scalar("Metrics/Top3_Acc", val_top3, epoch)
                self.writer.add_scalar("Metrics/ROC_AUC", val_auc, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # WandB logging
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "val_top1_acc": val_top1,
                    "val_top3_acc": val_top3,
                    "val_roc_auc": val_auc,
                    "learning_rate": current_lr,
                })

            # Check for best model
            is_better = (val_f1 > self.best_metric) if self.monitor_mode == "max" else (val_loss < self.best_metric)
            if is_better:
                self.best_metric = val_f1 if self.monitor_mode == "max" else val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best checkpoint
                best_path = self.run_dir / "best_model.pt"
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metric_val=self.best_metric,
                    class_to_idx=self.species_to_idx,
                    cfg=self.cfg,
                )
                logger.info(f"Saved new best model checkpoint to {best_path} (score: {self.best_metric:.4f}).")
            else:
                self.patience_counter += 1

            # Save last checkpoint
            last_path = self.run_dir / "last_model.pt"
            save_checkpoint(
                last_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metric_val=val_f1,
                class_to_idx=self.species_to_idx,
                cfg=self.cfg,
            )

            # Print epoch summary table
            row = [epoch, train_loss, val_loss, val_f1, val_top1, val_top3, val_auc, current_lr, epoch_time]
            history_rows.append(row)
            print("\n" + format_epoch_table(headers, [row]))

            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.early_stopping_patience} epochs "
                    f"without improvement (Best Epoch: {self.best_epoch}, Score: {self.best_metric:.4f})."
                )
                break

        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

        print("\n" + "=" * 60)
        print("TRAINING RUN COMPLETE")
        print("=" * 60)
        print(f"Best Epoch: {self.best_epoch} | Best {self.monitor_metric}: {self.best_metric:.4f}")
        print(f"Checkpoints and logs saved to: {self.run_dir}")

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "run_dir": str(self.run_dir),
        }

