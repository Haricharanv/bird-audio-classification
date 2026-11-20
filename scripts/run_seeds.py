"""Run experiment across multiple random seeds and report mean +/- std."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BirdDataset, create_dataloaders
from src.data.noise_bank import build_noise_bank, load_noise_bank
from src.data.preprocess import build_metadata
from src.data.splits import create_splits
from src.evaluation.metrics import compute_classification_metrics
from src.models.registry import build_model
from src.training.losses import build_loss
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.io import load_checkpoint, load_json
from src.utils.logging import format_epoch_table, get_logger
from src.utils.seed import seed_everything

logger = get_logger("run_seeds")


def run_single_seed(
    seed: int,
    cfg: Any,
    exp_dir: Path,
) -> dict[str, float]:
    """Train and evaluate a single seed."""
    logger.info("=" * 60)
    logger.info(f"STARTING SEED RUN: {seed}")
    logger.info("=" * 60)

    cfg.seed = seed
    seed_everything(seed, deterministic=cfg.deterministic)

    seed_run_dir = exp_dir / f"seed_{seed}"
    seed_run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, seed_run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_dir = Path(cfg.data.metadata_dir)
    splits_path = metadata_dir / "splits.csv"
    species_map_path = metadata_dir / "species_to_idx.json"
    noise_path = Path(cfg.data.noise_bank_path)

    if not splits_path.is_file() or not species_map_path.is_file():
        df, species_to_idx, _ = build_metadata(cfg)
        splits_df = create_splits(df, cfg)
        noise_bank = build_noise_bank(cfg)
    else:
        splits_df = pd.read_csv(splits_path, dtype={"primary_label": str, "filename": str})
        species_to_idx = load_json(species_map_path)
        noise_bank = load_noise_bank(noise_path) or build_noise_bank(cfg)

    num_classes = len(species_to_idx)
    cfg.model.num_classes = num_classes

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg=cfg,
        splits_df=splits_df,
        species_to_idx=species_to_idx,
        noise_bank=noise_bank,
    )

    # Model, Optimizer, Scheduler, Loss
    model = build_model(cfg, num_classes=num_classes).to(device)
    optimizer = build_optimizer(model, cfg)
    epochs = int(getattr(cfg.train, "epochs", 25))
    steps_per_epoch = len(train_loader) // max(1, int(getattr(cfg.train, "gradient_accumulation_steps", 1)))
    scheduler, step_mode = build_scheduler(optimizer, cfg, epochs, steps_per_epoch)

    train_counts = splits_df[splits_df["split"] == "train"]["primary_label"].value_counts().to_dict()
    criterion = build_loss(cfg, class_counts=train_counts, device=device)

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_mode=step_mode,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        species_to_idx=species_to_idx,
        run_dir=seed_run_dir,
    )
    trainer.fit()

    # Evaluate best model on held-out TEST split using sliding windows
    best_path = seed_run_dir / "best_model.pt"
    load_checkpoint(best_path, model=model, device=device)
    model.eval()

    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    test_dataset = BirdDataset(
        df=test_df,
        cfg=cfg,
        split="test",
        species_to_idx=species_to_idx,
        use_sliding_windows=True,
    )

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            w = sample["waveform"].to(device)
            target = sample["target"].numpy()
            logits = model(w)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(np.mean(probs, axis=0))
            all_targets.append(target)

    y_pred_probs = np.stack(all_preds, axis=0)
    y_true = np.array(all_targets)

    idx_to_sp = {v: k for k, v in species_to_idx.items()}
    class_names = [idx_to_sp[i] for i in range(len(species_to_idx))]
    metrics = compute_classification_metrics(y_true, y_pred_probs, class_names=class_names)
    metrics.pop("per_class_report", None)

    return {
        "seed": seed,
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "top1_acc": metrics["top1_accuracy"],
        "top3_acc": metrics["top3_accuracy"],
        "roc_auc": metrics["roc_auc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple seeds for an experiment.")
    parser.add_argument("--config", type=str, default="configs/baseline_cnn.yaml", help="Base config path.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate.")
    parser.add_argument("overrides", nargs="*", help="CLI overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = str(getattr(cfg, "experiment_name", "run_seeds"))
    reports_dir = Path("reports") / "seeds" / f"{timestamp}_{exp_name}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.seeds:
        res = run_single_seed(s, cfg, reports_dir)
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / "seeds_metrics.csv", index=False)

    # Compute mean and standard deviation
    metrics_cols = ["macro_f1", "weighted_f1", "top1_acc", "top3_acc", "roc_auc"]
    means = results_df[metrics_cols].mean()
    stds = results_df[metrics_cols].std()

    summary_headers = ["Metric", "Mean", "Std", "Seeds Results (Mean +/- Std)"]
    summary_rows = []
    for col in metrics_cols:
        m = means[col]
        s = stds[col]
        summary_rows.append([col, f"{m:.4f}", f"{s:.4f}", f"{m:.4f} +/- {s:.4f}"])

    print("\n" + "=" * 60)
    print(f"MULTI-SEED EVALUATION SUMMARY ({len(args.seeds)} SEEDS: {args.seeds})")
    print("=" * 60)
    print(format_epoch_table(summary_headers, summary_rows))

    summary_df = pd.DataFrame({
        "metric": metrics_cols,
        "mean": [means[c] for c in metrics_cols],
        "std": [stds[c] for c in metrics_cols],
    })
    summary_df.to_csv(reports_dir / "seeds_summary.csv", index=False)
    logger.info(f"Saved seed experiment summary to {reports_dir / 'seeds_summary.csv'}.")


if __name__ == "__main__":
    main()

