"""CLI training script for Bird Audio Classification supporting CLI dot overrides and experiment tracking."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.data.noise_bank import build_noise_bank, load_noise_bank
from src.data.preprocess import build_metadata, cache_energy_chunks
from src.data.splits import create_splits
from src.models.registry import build_model
from src.training.losses import build_loss
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.io import load_json
from src.utils.logging import get_logger
from src.utils.seed import seed_everything

logger = get_logger("train")


def resolve_device(device_str: str) -> torch.device:
    """Resolve compute device from string (auto, cuda, mps, cpu)."""
    device_lower = device_str.lower()
    if device_lower == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_lower)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bird audio classification model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base YAML config.")
    parser.add_argument("overrides", nargs="*", help="Key=value overrides (e.g. model.backbone=resnet34 debug=true).")
    args = parser.parse_args()

    # 1. Load hierarchical config and apply dot overrides
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    # 2. Setup run directory: runs/<timestamp>_<experiment_name>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = str(getattr(cfg, "experiment_name", "experiment"))
    run_dir = Path("runs") / f"{timestamp}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    # 3. Resolve device
    device_cfg = str(getattr(cfg.system, "device", "auto"))
    device = resolve_device(device_cfg)
    logger.info("=" * 70)
    logger.info(f"Training Run: {run_dir.name}")
    logger.info(f"Experiment:   {exp_name}")
    logger.info(f"Target Device: {device}")
    logger.info(f"Debug Mode:   {cfg.debug}")
    logger.info("=" * 70)

    # 4. Prepare data: ensure metadata, splits, and noise bank exist
    metadata_dir = Path(cfg.data.metadata_dir)
    splits_path = metadata_dir / "splits.csv"
    species_map_path = metadata_dir / "species_to_idx.json"
    noise_path = Path(cfg.data.noise_bank_path)

    if not splits_path.is_file() or not species_map_path.is_file():
        logger.info("Splits or species map missing; triggering data preparation...")
        df, species_to_idx, _ = build_metadata(cfg)
        splits_df = create_splits(df, cfg)
        noise_bank = build_noise_bank(cfg)
    else:
        splits_df = pd.read_csv(splits_path, dtype={"primary_label": str, "filename": str})
        species_to_idx = load_json(species_map_path)
        noise_bank = load_noise_bank(noise_path)
        if noise_bank is None:
            noise_bank = build_noise_bank(cfg)

    # In debug mode: limit epochs if not explicitly overridden
    if cfg.debug and getattr(cfg.train, "epochs", 25) > 1 and not any("train.epochs=" in o for o in args.overrides):
        cfg.train.epochs = 1
        logger.info("DEBUG MODE: Setting train.epochs = 1 for CPU-friendly test run.")

    # Check for pre-cached chunks catalog
    cached_chunks_path = metadata_dir / "cached_chunks.csv"
    cached_chunks_df = pd.read_csv(cached_chunks_path) if cached_chunks_path.is_file() else None

    # 5. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg=cfg,
        splits_df=splits_df,
        species_to_idx=species_to_idx,
        noise_bank=noise_bank,
        cached_chunks_df=cached_chunks_df,
    )

    num_classes = len(species_to_idx)
    cfg.model.num_classes = num_classes

    # 6. Build Model
    model = build_model(cfg, num_classes=num_classes)
    model.to(device)

    # 7. Build Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg)
    epochs = int(getattr(cfg.train, "epochs", 25))
    steps_per_epoch = len(train_loader) // max(1, int(getattr(cfg.train, "gradient_accumulation_steps", 1)))
    scheduler, scheduler_step_mode = build_scheduler(optimizer, cfg, epochs, steps_per_epoch)

    # 8. Build Loss Criterion
    train_counts = splits_df[splits_df["split"] == "train"]["primary_label"].value_counts().to_dict()
    criterion = build_loss(cfg, class_counts=train_counts, device=device)

    # 9. Train Model
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_mode=scheduler_step_mode,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        species_to_idx=species_to_idx,
        run_dir=run_dir,
    )

    results = trainer.fit()
    logger.info(f"Training successfully completed. Best metric: {results['best_metric']:.4f}")


if __name__ == "__main__":
    main()

