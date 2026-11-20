"""Data processing, splits, dataset, noise bank, and augmentation package."""

from src.data.preprocess import (
    load_audio,
    compute_chunk_rms,
    extract_energy_chunks,
    build_metadata,
    cache_energy_chunks,
    parse_secondary_labels,
)
from src.data.splits import create_splits, split_by_filename, split_by_author, sanitize_author_groups
from src.data.noise_bank import build_noise_bank, load_noise_bank
from src.data.augment import WaveformAugmenter, SpecAugment, BatchMixupCutmix
from src.data.dataset import BirdDataset, create_dataloaders

__all__ = [
    "load_audio",
    "compute_chunk_rms",
    "extract_energy_chunks",
    "build_metadata",
    "cache_energy_chunks",
    "parse_secondary_labels",
    "create_splits",
    "split_by_filename",
    "split_by_author",
    "sanitize_author_groups",
    "build_noise_bank",
    "load_noise_bank",
    "WaveformAugmenter",
    "SpecAugment",
    "BatchMixupCutmix",
    "BirdDataset",
    "create_dataloaders",
]

