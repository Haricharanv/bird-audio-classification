"""CLI script to build metadata, generate leakage-free splits, construct noise bank, and cache chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.noise_bank import build_noise_bank
from src.data.preprocess import build_metadata, cache_energy_chunks
from src.data.splits import create_splits
from src.utils.config import load_config
from src.utils.logging import format_epoch_table, get_logger
from src.utils.seed import seed_everything

logger = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata, splits, noise bank, and cached chunks.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base YAML config.")
    parser.add_argument("overrides", nargs="*", help="Key=value configuration overrides (e.g. debug=true).")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    logger.info("=" * 70)
    logger.info(f"Starting Data Preparation Pipeline (Debug={cfg.debug})")
    logger.info(f"Raw Data Directory: {cfg.data.raw_dir}")
    logger.info(f"Metadata Directory: {cfg.data.metadata_dir}")
    logger.info(f"Cache Directory:    {cfg.data.cache_dir}")
    logger.info("=" * 70)

    # 1. Ingest, harmonize, and filter metadata
    logger.info("[1/4] Ingesting and filtering metadata...")
    df, species_to_idx, dropped_species = build_metadata(cfg)

    # 2. Construct leakage-free train/val/test splits
    logger.info(f"[2/4] Generating leakage-free splits using strategy '{cfg.data.split_by}'...")
    splits_df = create_splits(df, cfg)

    # 3. Build background soundscape noise bank
    logger.info(f"[3/4] Building noise bank ({cfg.data.noise_bank_size} samples)...")
    noise_bank = build_noise_bank(cfg)

    # 4. Pre-chunk audio and cache high-energy segments
    logger.info("[4/4] Pre-chunking audio and caching top-energy float16 segments...")
    cached_chunks_df = cache_energy_chunks(splits_df, cfg)

    # Print Summary Table
    total_recs = len(splits_df)
    train_count = (splits_df["split"] == "train").sum()
    val_count = (splits_df["split"] == "val").sum()
    test_count = (splits_df["split"] == "test").sum()

    split_headers = ["Split", "Recordings", "Percentage", "Species Count"]
    split_rows = [
        ["Train", train_count, f"{train_count/total_recs:.1%}", splits_df[splits_df["split"]=="train"]["primary_label"].nunique()],
        ["Val", val_count, f"{val_count/total_recs:.1%}", splits_df[splits_df["split"]=="val"]["primary_label"].nunique()],
        ["Test", test_count, f"{test_count/total_recs:.1%}", splits_df[splits_df["split"]=="test"]["primary_label"].nunique()],
        ["Total", total_recs, "100.0%", len(species_to_idx)],
    ]
    print("\n" + "=" * 50)
    print("SPLIT DISTRIBUTION SUMMARY")
    print("=" * 50)
    print(format_epoch_table(split_headers, split_rows))

    # Collection breakdown
    coll_counts = splits_df["collection"].value_counts()
    coll_headers = ["Collection", "Recordings", "Percentage"]
    coll_rows = [[coll, count, f"{count/total_recs:.1%}"] for coll, count in coll_counts.items()]
    print("\n" + "=" * 50)
    print("COLLECTION SOURCE SUMMARY")
    print("=" * 50)
    print(format_epoch_table(coll_headers, coll_rows))
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()

