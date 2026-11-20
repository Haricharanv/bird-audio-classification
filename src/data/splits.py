"""Leakage-free dataset splitting by filename or author with stratification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger

logger = get_logger("bird_classification.splits")


def sanitize_author_groups(df: pd.DataFrame) -> pd.Series:
    """Assign distinct unique synthetic IDs to missing, empty, or 'Unknown' authors.
    
    This prevents disparate anonymous recordists from collapsing into a single mega-cluster.
    
    Args:
        df: Input DataFrame containing 'author' column.
        
    Returns:
        Series of sanitized author group IDs.
    """
    authors = df["author"].fillna("Unknown").astype(str).str.strip()
    sanitized = authors.copy()
    
    unknown_mask = sanitized.str.lower().isin(["unknown", "nan", "none", "", "null"])
    unknown_indices = df[unknown_mask].index
    for idx in unknown_indices:
        sanitized.loc[idx] = f"Unknown_Recordist_{idx}"
        
    return sanitized


def split_by_filename(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified split by filename.
    
    Species with fewer than 3 recordings are placed into 'train' only to ensure validity.
    
    Args:
        df: Metadata DataFrame where each row is a unique recording filename.
        train_ratio: Fraction for train split (e.g. 0.70).
        val_ratio: Fraction for val split (e.g. 0.15).
        test_ratio: Fraction for test split (e.g. 0.15).
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with an added 'split' column ('train', 'val', 'test').
    """
    df = df.copy()
    df["split"] = "train"

    # Identify species counts
    species_counts = df["primary_label"].value_counts()
    rare_species = species_counts[species_counts < 3].index
    stratifiable_species = species_counts[species_counts >= 3].index

    logger.info(
        f"Splitting by filename: {len(stratifiable_species)} stratifiable species (>= 3 files), "
        f"{len(rare_species)} rare species (< 3 files placed train-only)."
    )

    strat_mask = df["primary_label"].isin(stratifiable_species)
    strat_df = df[strat_mask]

    temp_ratio = val_ratio + test_ratio  # e.g. 0.30
    val_in_temp_ratio = val_ratio / temp_ratio  # e.g. 0.50

    # First split: Train vs (Val + Test)
    train_indices, temp_indices = train_test_split(
        strat_df.index,
        test_size=temp_ratio,
        stratify=strat_df["primary_label"],
        random_state=seed,
    )

    # Second split: Val vs Test
    temp_df = df.loc[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_df.index,
        test_size=1.0 - val_in_temp_ratio,
        stratify=temp_df["primary_label"],
        random_state=seed,
    )

    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "val"
    df.loc[test_indices, "split"] = "test"

    return df


def split_by_author(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Group-aware disjoint split by recordist (author).
    
    Guarantees no author spans more than one split, preventing recordist acoustic leakage.
    Unknown authors are treated as individual unique groups.
    
    Args:
        df: Metadata DataFrame.
        train_ratio: Fraction for train split.
        val_ratio: Fraction for val split.
        test_ratio: Fraction for test split.
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with an added 'split' column ('train', 'val', 'test').
    """
    df = df.copy()
    author_groups = sanitize_author_groups(df)
    df["_author_group"] = author_groups

    # Build author summary: total recordings and species present
    rng = np.random.RandomState(seed)
    unique_authors = list(df["_author_group"].unique())
    rng.shuffle(unique_authors)

    # Sort authors by total recording count descending for greedy bin packing
    author_counts = df["_author_group"].value_counts()
    sorted_authors = author_counts.index.tolist()

    total_files = len(df)
    target_train = int(total_files * train_ratio)
    target_val = int(total_files * val_ratio)
    target_test = total_files - target_train - target_val

    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": target_train, "val": target_val, "test": target_test}
    author_to_split: Dict[str, str] = {}

    for auth in sorted_authors:
        auth_len = author_counts[auth]
        # Find which partition has the greatest remaining need
        remaining = {s: targets[s] - counts[s] for s in ["train", "val", "test"]}
        best_split = max(remaining, key=lambda s: remaining[s])
        
        author_to_split[auth] = best_split
        counts[best_split] += auth_len

    df["split"] = df["_author_group"].map(author_to_split)
    df.drop(columns=["_author_group"], inplace=True)
    return df


def create_splits(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Create and persist leakage-free train, val, and test splits.
    
    Args:
        df: Metadata DataFrame.
        cfg: Configuration object.
        
    Returns:
        DataFrame with 'split' column saved to data/metadata/splits.csv.
    """
    split_strategy = str(getattr(cfg.data, "split_by", "filename")).lower()
    train_r = float(getattr(cfg.data, "train_ratio", 0.70))
    val_r = float(getattr(cfg.data, "val_ratio", 0.15))
    test_r = float(getattr(cfg.data, "test_ratio", 0.15))
    seed = int(getattr(cfg, "seed", 42))

    if split_strategy == "author":
        splits_df = split_by_author(
            df,
            train_ratio=train_r,
            val_ratio=val_r,
            test_ratio=test_r,
            seed=seed,
        )
    elif split_strategy == "filename":
        splits_df = split_by_filename(
            df,
            train_ratio=train_r,
            val_ratio=val_r,
            test_ratio=test_r,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown split_by mode: '{split_strategy}'. Choose 'filename' or 'author'.")

    # Verification: assert filenames are unique across splits
    train_files = set(splits_df[splits_df["split"] == "train"]["filename"])
    val_files = set(splits_df[splits_df["split"] == "val"]["filename"])
    test_files = set(splits_df[splits_df["split"] == "test"]["filename"])

    assert len(train_files.intersection(val_files)) == 0, "Data leakage detected: train and val share filenames!"
    assert len(train_files.intersection(test_files)) == 0, "Data leakage detected: train and test share filenames!"
    assert len(val_files.intersection(test_files)) == 0, "Data leakage detected: val and test share filenames!"

    # In author mode, verify authors are completely disjoint
    if split_strategy == "author":
        sanitized_authors = sanitize_author_groups(splits_df)
        train_authors = set(sanitized_authors[splits_df["split"] == "train"])
        val_authors = set(sanitized_authors[splits_df["split"] == "val"])
        test_authors = set(sanitized_authors[splits_df["split"] == "test"])
        assert len(train_authors.intersection(val_authors)) == 0, "Author leakage between train and val!"
        assert len(train_authors.intersection(test_authors)) == 0, "Author leakage between train and test!"
        assert len(val_authors.intersection(test_authors)) == 0, "Author leakage between val and test!"

    # Save to data/metadata/splits.csv
    metadata_dir = Path(cfg.data.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_path = metadata_dir / "splits.csv"
    splits_df.to_csv(splits_path, index=False)

    # Log split distribution
    counts = splits_df["split"].value_counts().to_dict()
    total = len(splits_df)
    logger.info(
        f"Splits generated ({split_strategy}): Train={counts.get('train', 0)} ({counts.get('train', 0)/total:.1%}), "
        f"Val={counts.get('val', 0)} ({counts.get('val', 0)/total:.1%}), "
        f"Test={counts.get('test', 0)} ({counts.get('test', 0)/total:.1%})."
    )

    return splits_df

