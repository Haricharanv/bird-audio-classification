"""Systematic ablation runner driven by configs/ablations.yaml."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BirdDataset, create_dataloaders
from src.data.noise_bank import build_noise_bank, load_noise_bank
from src.data.preprocess import build_metadata
from src.data.splits import create_splits
from src.models.registry import build_model
from src.training.losses import build_loss
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import Config, _deep_merge, load_config
from src.utils.io import load_json
from src.utils.logging import format_epoch_table, get_logger
from src.utils.seed import seed_everything

logger = get_logger("ablations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run systematic ablation studies.")
    parser.add_argument("--config", type=str, default="configs/ablations.yaml", help="Path to ablations YAML.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Epoch limit override per ablation experiment.")
    parser.add_argument("cli_overrides", nargs="*", help="Global CLI overrides (e.g. debug=true).")
    args = parser.parse_args()

    ablation_cfg_path = Path(args.config)
    with open(ablation_cfg_path, "r", encoding="utf-8") as f:
        ablation_raw = yaml.safe_load(f)

    base_config_path = ablation_raw.get("base_config", "configs/effnet_b0.yaml")
    experiments = ablation_raw.get("experiments", [])
    output_dir = Path(ablation_raw.get("output_dir", "reports/ablations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Loaded {len(experiments)} ablation experiments from {ablation_cfg_path}")
    logger.info(f"Base Config: {base_config_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_records: list[dict[str, Any]] = []

    for exp_idx, exp in enumerate(experiments, start=1):
        exp_name = exp.get("name", f"exp_{exp_idx}")
        overrides_dict = exp.get("overrides", {})
        logger.info(f"\n[{exp_idx}/{len(experiments)}] Running Ablation: '{exp_name}'...")

        # Flatten overrides_dict into CLI list: ['audio.feature_type=mfcc', ...]
        override_list = [f"{k}={v}" for k, v in overrides_dict.items()]
        if args.cli_overrides:
            override_list.extend(args.cli_overrides)

        cfg = load_config(base_config_path, override_list)
        seed_everything(cfg.seed, deterministic=cfg.deterministic)

        if args.max_epochs is not None:
            cfg.train.epochs = args.max_epochs
        elif cfg.debug:
            cfg.train.epochs = 1

        # Check data
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

        train_loader, val_loader, _ = create_dataloaders(
            cfg=cfg,
            splits_df=splits_df,
            species_to_idx=species_to_idx,
            noise_bank=noise_bank,
        )

        model = build_model(cfg, num_classes=num_classes).to(device)
        optimizer = build_optimizer(model, cfg)
        epochs = int(getattr(cfg.train, "epochs", 1))
        steps_per_epoch = len(train_loader) // max(1, int(getattr(cfg.train, "gradient_accumulation_steps", 1)))
        scheduler, step_mode = build_scheduler(optimizer, cfg, epochs, steps_per_epoch)

        train_counts = splits_df[splits_df["split"] == "train"]["primary_label"].value_counts().to_dict()
        criterion = build_loss(cfg, class_counts=train_counts, device=device)

        run_sub_dir = output_dir / exp_name
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
            run_dir=run_sub_dir,
        )

        res = trainer.fit()
        val_metrics = trainer.validate()

        record = {
            "experiment": exp_name,
            "val_macro_f1": val_metrics["val_macro_f1"],
            "val_top1_acc": val_metrics["val_top1_acc"],
            "val_top3_acc": val_metrics["val_top3_acc"],
            "val_roc_auc": val_metrics["val_roc_auc"],
            "val_loss": val_metrics["val_loss"],
            "best_epoch": res["best_epoch"],
        }
        results_records.append(record)
        logger.info(f"Finished {exp_name} -> Macro-F1: {val_metrics['val_macro_f1']:.4f}")

    # Generate Consolidated Comparison Table
    df_results = pd.DataFrame(results_records)
    csv_path = output_dir / "ablation_summary.csv"
    md_path = output_dir / "ablation_summary.md"

    df_results.to_csv(csv_path, index=False)

    table_headers = ["Experiment", "Val Macro-F1", "Top-1 Acc", "Top-3 Acc", "ROC-AUC", "Val Loss", "Best Epoch"]
    table_rows = [
        [
            r["experiment"],
            r["val_macro_f1"],
            r["val_top1_acc"],
            r["val_top3_acc"],
            r["val_roc_auc"],
            r["val_loss"],
            r["best_epoch"],
        ]
        for r in results_records
    ]

    md_table = format_epoch_table(table_headers, table_rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Systematic Ablation Study Results\n\n")
        f.write(md_table + "\n")

    print("\n" + "=" * 70)
    print("CONSOLIDATED ABLATION STUDY RESULTS")
    print("=" * 70)
    print(md_table)
    print(f"\nSaved ablation summary to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()

