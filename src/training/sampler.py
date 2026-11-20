"""Class imbalance sampling utilities: WeightedRandomSampler implementation."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

from src.utils.logging import get_logger

logger = get_logger("bird_classification.sampler")


def build_class_weighted_sampler(
    df: pd.DataFrame,
    species_col: str = "primary_label",
    power: float = 0.5,
) -> WeightedRandomSampler:
    """Build a PyTorch WeightedRandomSampler balancing rare and frequent bird species.
    
    Weights each sample by 1 / (class_count ** power).
    power=0.5 corresponds to inverse square root weighting (smoothing aggressive rebalancing),
    while power=1.0 is inverse frequency weighting.
    
    Args:
        df: Training DataFrame containing species labels.
        species_col: Column name containing species identifiers.
        power: Exponent for frequency attenuation (default 0.5 = inverse sqrt).
        
    Returns:
        Instantiated WeightedRandomSampler.
    """
    class_counts = df[species_col].value_counts().to_dict()
    weights_per_class = {sp: 1.0 / (count**power) for sp, count in class_counts.items()}
    sample_weights = [weights_per_class[sp] for sp in df[species_col]]

    logger.info(
        f"Built WeightedRandomSampler across {len(class_counts)} classes "
        f"(power={power}, min_class_count={min(class_counts.values())}, "
        f"max_class_count={max(class_counts.values())})."
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

