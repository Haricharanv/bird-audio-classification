"""PyTorch Dataset and DataLoader implementations for bird bioacoustic classification."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.augment import WaveformAugmenter
from src.data.preprocess import load_audio
from src.utils.logging import get_logger
from src.utils.seed import worker_init_fn

logger = get_logger("bird_classification.dataset")


class BirdDataset(Dataset):
    """PyTorch Dataset for BirdCLEF+ 2025 audio classification.
    
    Supports training with random 5s crops and waveform augmentations,
    as well as validation/testing with sliding 5s windows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Any,
        split: str = "train",
        species_to_idx: Optional[Dict[str, int]] = None,
        noise_bank: Optional[np.ndarray] = None,
        cached_chunks_df: Optional[pd.DataFrame] = None,
        use_sliding_windows: bool = False,
    ):
        """Initialize BirdDataset.
        
        Args:
            df: Filtered metadata DataFrame containing 'filename', 'primary_label', etc.
            cfg: Configuration object.
            split: 'train', 'val', or 'test'.
            species_to_idx: Dictionary mapping species labels to class indices.
            noise_bank: Pre-loaded background soundscape noise array.
            cached_chunks_df: Optional DataFrame pointing to pre-cached float16 .npy chunks.
            use_sliding_windows: If True (for val/test full recording evaluation),
                                 returns all sliding 5s windows for a recording.
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.split = split
        self.species_to_idx = species_to_idx or {}
        self.num_classes = len(self.species_to_idx)
        self.use_sliding_windows = use_sliding_windows
        self.is_multilabel = str(getattr(cfg, "task", "multiclass")).lower() == "multilabel"

        self.sr = int(getattr(cfg.audio, "sample_rate", 32000))
        self.duration_sec = float(getattr(cfg.audio, "duration_sec", 5.0))
        self.target_samples = int(self.duration_sec * self.sr)
        self.hop_samples = int(float(getattr(cfg.audio, "val_hop_sec", 2.5)) * self.sr)

        self.raw_dir = Path(cfg.data.raw_dir)
        self.audio_dir = self.raw_dir / "train_audio"
        self.cache_dir = Path(cfg.data.cache_dir)

        # Build chunk lookup index if cached chunks provided
        self.has_cached_chunks = False
        if cached_chunks_df is not None and not cached_chunks_df.empty:
            self.cached_chunks = cached_chunks_df[
                cached_chunks_df["filename"].isin(self.df["filename"])
            ].reset_index(drop=True)
            self.has_cached_chunks = len(self.cached_chunks) > 0

        # Augmentation pipeline for training
        if self.split == "train":
            self.augmenter = WaveformAugmenter(cfg, noise_bank=noise_bank)
        else:
            self.augmenter = None

    def __len__(self) -> int:
        if self.split == "train" and self.has_cached_chunks:
            return len(self.cached_chunks)
        return len(self.df)

    def _resolve_audio_path(self, filename: str, species: str) -> Path:
        """Find audio file in raw directory."""
        direct = self.audio_dir / filename
        if direct.is_file():
            return direct
        sub = self.audio_dir / species / filename
        if sub.is_file():
            return sub
        return direct

    def _pad_or_crop(self, waveform: np.ndarray, is_random: bool = True) -> np.ndarray:
        """Ensure waveform is exactly target_samples in length."""
        total_len = len(waveform)
        if total_len == self.target_samples:
            return waveform

        if total_len < self.target_samples:
            if total_len == 0:
                return np.zeros(self.target_samples, dtype=np.float32)
            repeats = int(np.ceil(self.target_samples / total_len))
            return np.tile(waveform, repeats)[: self.target_samples]

        # Crop to target_samples
        max_start = total_len - self.target_samples
        if is_random:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2  # Center crop for deterministic validation
        return waveform[start : start + self.target_samples]

    def _get_sliding_windows(self, waveform: np.ndarray) -> np.ndarray:
        """Extract sliding 5s windows with 2.5s hop for full recording evaluation."""
        total_len = len(waveform)
        if total_len <= self.target_samples:
            return self._pad_or_crop(waveform, is_random=False)[np.newaxis, :]

        windows = []
        for start in range(0, total_len - self.target_samples + 1, self.hop_samples):
            windows.append(waveform[start : start + self.target_samples])

        tail_start = total_len - self.target_samples
        if tail_start > 0 and (not windows or (total_len - self.target_samples) % self.hop_samples != 0):
            windows.append(waveform[tail_start:total_len])

        return np.stack(windows, axis=0)  # (num_windows, target_samples)

    def _build_target(self, row: pd.Series) -> torch.Tensor:
        """Construct integer class index (multiclass) or multi-hot vector (multilabel)."""
        primary = str(row["primary_label"])
        target_idx = self.species_to_idx.get(primary, 0)

        if not self.is_multilabel:
            return torch.tensor(target_idx, dtype=torch.long)

        # Multilabel: multi-hot vector
        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        if primary in self.species_to_idx:
            multi_hot[target_idx] = 1.0

        secondaries = row.get("secondary_labels", [])
        if isinstance(secondaries, list):
            for sec in secondaries:
                sec_str = str(sec).strip()
                if sec_str in self.species_to_idx:
                    multi_hot[self.species_to_idx[sec_str]] = 1.0

        return multi_hot

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch item by index.
        
        Returns:
            Dict containing:
                'waveform': Tensor (target_samples,) or (num_windows, target_samples)
                'target': Tensor class index or multi-hot vector
                'filename': str
                'primary_label': str
                'collection': str
                'author': str
        """
        # If loading from cached float16 chunks in training
        if self.split == "train" and self.has_cached_chunks:
            row = self.cached_chunks.iloc[idx]
            chunk_rel_path = row["chunk_path"]
            chunk_full_path = self.cache_dir.parent / chunk_rel_path
            
            try:
                waveform = np.load(str(chunk_full_path)).astype(np.float32)
            except Exception as e:
                logger.warning(f"Error reading chunk {chunk_full_path}: {e}; fallback to silence.")
                waveform = np.zeros(self.target_samples, dtype=np.float32)

            if len(waveform) != self.target_samples:
                waveform = self._pad_or_crop(waveform, is_random=True)

            if self.augmenter is not None:
                waveform = self.augmenter(waveform)

            target = self._build_target(row)
            return {
                "waveform": torch.from_numpy(waveform).float(),
                "target": target,
                "filename": str(row["filename"]),
                "primary_label": str(row["primary_label"]),
                "collection": str(row.get("collection", "XC")),
                "author": str(row.get("author", "Unknown")),
            }

        # Otherwise loading from audio file directly
        row = self.df.iloc[idx]
        species = str(row["primary_label"])
        filename = str(row["filename"])
        audio_path = self._resolve_audio_path(filename, species)

        try:
            waveform = load_audio(
                audio_path,
                target_sr=self.sr,
                trim_silence=bool(getattr(self.cfg.audio, "trim_silence", True)),
                trim_top_db=float(getattr(self.cfg.audio, "trim_top_db", 30.0)),
                peak_norm=bool(getattr(self.cfg.audio, "peak_normalize", True)),
            )
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            waveform = np.zeros(self.target_samples, dtype=np.float32)

        target = self._build_target(row)

        if self.use_sliding_windows:
            windows = self._get_sliding_windows(waveform)
            return {
                "waveform": torch.from_numpy(windows).float(),  # (num_windows, target_samples)
                "target": target,
                "filename": filename,
                "primary_label": species,
                "collection": str(row.get("collection", "XC")),
                "author": str(row.get("author", "Unknown")),
            }

        is_random = (self.split == "train")
        waveform = self._pad_or_crop(waveform, is_random=is_random)

        if self.split == "train" and self.augmenter is not None:
            waveform = self.augmenter(waveform)

        return {
            "waveform": torch.from_numpy(waveform).float(),
            "target": target,
            "filename": filename,
            "primary_label": species,
            "collection": str(row.get("collection", "XC")),
            "author": str(row.get("author", "Unknown")),
        }


def create_dataloaders(
    cfg: Any,
    splits_df: pd.DataFrame,
    species_to_idx: Dict[str, int],
    noise_bank: Optional[np.ndarray] = None,
    cached_chunks_df: Optional[pd.DataFrame] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiate PyTorch DataLoaders for train, val, and test splits.
    
    Args:
        cfg: Configuration object.
        splits_df: DataFrame containing 'split' column ('train', 'val', 'test').
        species_to_idx: Mapping from species to integer class index.
        noise_bank: Pre-loaded noise bank array.
        cached_chunks_df: Optional pre-cached chunks catalog.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    batch_size = int(getattr(cfg.train, "batch_size", 32))
    num_workers = int(getattr(cfg.system, "num_workers", 4))
    pin_memory = bool(getattr(cfg.system, "pin_memory", True)) and torch.cuda.is_available()

    train_dataset = BirdDataset(
        df=train_df,
        cfg=cfg,
        split="train",
        species_to_idx=species_to_idx,
        noise_bank=noise_bank,
        cached_chunks_df=cached_chunks_df,
        use_sliding_windows=False,
    )

    val_dataset = BirdDataset(
        df=val_df,
        cfg=cfg,
        split="val",
        species_to_idx=species_to_idx,
        use_sliding_windows=False,
    )

    test_dataset = BirdDataset(
        df=test_df,
        cfg=cfg,
        split="test",
        species_to_idx=species_to_idx,
        use_sliding_windows=False,
    )

    # Optional WeightedRandomSampler for class imbalance handling
    use_weighted_sampler = bool(getattr(cfg.train, "use_weighted_sampler", False))
    sampler = None
    shuffle = True

    if use_weighted_sampler:
        class_counts = train_df["primary_label"].value_counts()
        weights_per_class = {sp: 1.0 / np.sqrt(count) for sp, count in class_counts.items()}
        sample_weights = [weights_per_class[sp] for sp in train_df["primary_label"]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    logger.info(
        f"DataLoaders created: Train batches={len(train_loader)}, "
        f"Val batches={len(val_loader)}, Test batches={len(test_loader)}."
    )

    return train_loader, val_loader, test_loader

