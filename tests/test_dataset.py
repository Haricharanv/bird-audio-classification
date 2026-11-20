"""Unit tests for BirdDataset, waveform augmentations, padding, and target generation."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.augment import WaveformAugmenter
from src.data.dataset import BirdDataset
from src.utils.config import Config


@pytest.fixture
def dummy_cfg():
    return Config({
        "task": "multiclass",
        "data": {
            "raw_dir": "/tmp/dummy_raw",
            "cache_dir": "/tmp/dummy_cache",
            "metadata_dir": "/tmp/dummy_meta",
        },
        "audio": {
            "sample_rate": 32000,
            "duration_sec": 5.0,
            "val_hop_sec": 2.5,
            "trim_silence": False,
            "peak_normalize": True,
        },
        "augment": {
            "waveform": {
                "noise_bank_prob": 1.0,
                "noise_snr_min_db": 5.0,
                "noise_snr_max_db": 15.0,
                "gain_prob": 1.0,
                "gain_min_db": -3.0,
                "gain_max_db": 3.0,
                "pitch_shift_prob": 0.0,
                "time_stretch_prob": 0.0,
                "time_shift_prob": 1.0,
            }
        },
    })


@pytest.fixture
def dummy_metadata():
    return pd.DataFrame([
        {
            "filename": "bird_01.ogg",
            "primary_label": "sp_a",
            "secondary_labels": ["sp_b"],
            "author": "Alice",
            "collection": "XC",
            "target": 0,
        },
        {
            "filename": "bird_02.ogg",
            "primary_label": "sp_b",
            "secondary_labels": [],
            "author": "Bob",
            "collection": "iNat",
            "target": 1,
        },
    ])


def test_pad_or_crop_short_and_long_audio(dummy_cfg, dummy_metadata):
    species_to_idx = {"sp_a": 0, "sp_b": 1}
    dataset = BirdDataset(
        df=dummy_metadata,
        cfg=dummy_cfg,
        split="val",
        species_to_idx=species_to_idx,
    )
    
    # 1. Short audio (1 second = 32000 samples) -> must repeat-pad to 160,000
    short_audio = np.ones(32000, dtype=np.float32) * 0.5
    padded = dataset._pad_or_crop(short_audio, is_random=False)
    assert len(padded) == 160000
    assert not np.isnan(padded).any()

    # 2. Long audio (8 seconds = 256000 samples) -> must crop to 160,000
    long_audio = np.random.randn(256000).astype(np.float32)
    cropped = dataset._pad_or_crop(long_audio, is_random=False)
    assert len(cropped) == 160000


def test_sliding_windows_generation(dummy_cfg, dummy_metadata):
    species_to_idx = {"sp_a": 0, "sp_b": 1}
    dataset = BirdDataset(
        df=dummy_metadata,
        cfg=dummy_cfg,
        split="val",
        species_to_idx=species_to_idx,
        use_sliding_windows=True,
    )

    # 10 seconds audio @ 32 kHz = 320,000 samples
    # 5s window = 160,000 samples, 2.5s hop = 80,000 samples
    # Windows start at: 0, 80000, 160000 -> 3 windows
    audio_10s = np.random.randn(320000).astype(np.float32)
    windows = dataset._get_sliding_windows(audio_10s)
    
    assert windows.ndim == 2
    assert windows.shape[0] == 3
    assert windows.shape[1] == 160000


def test_multiclass_vs_multilabel_targets(dummy_cfg, dummy_metadata):
    species_to_idx = {"sp_a": 0, "sp_b": 1, "sp_c": 2}

    # 1. Multiclass mode
    dummy_cfg.task = "multiclass"
    dataset_mc = BirdDataset(
        df=dummy_metadata,
        cfg=dummy_cfg,
        split="val",
        species_to_idx=species_to_idx,
    )
    target_mc = dataset_mc._build_target(dummy_metadata.iloc[0])
    assert target_mc.dim() == 0  # scalar
    assert target_mc.dtype == torch.long
    assert target_mc.item() == 0

    # 2. Multilabel mode (primary: sp_a, secondary: [sp_b])
    dummy_cfg.task = "multilabel"
    dataset_ml = BirdDataset(
        df=dummy_metadata,
        cfg=dummy_cfg,
        split="val",
        species_to_idx=species_to_idx,
    )
    target_ml = dataset_ml._build_target(dummy_metadata.iloc[0])
    assert target_ml.shape == (3,)
    assert target_ml.dtype == torch.float32
    assert target_ml[0].item() == 1.0  # sp_a
    assert target_ml[1].item() == 1.0  # sp_b
    assert target_ml[2].item() == 0.0  # sp_c


def test_waveform_augmenter_with_noise_bank(dummy_cfg):
    target_samples = 160000
    noise_bank = np.random.randn(5, target_samples).astype(np.float32) * 0.1
    augmenter = WaveformAugmenter(dummy_cfg, noise_bank=noise_bank)

    clean_audio = np.random.randn(target_samples).astype(np.float32) * 0.5
    augmented = augmenter(clean_audio)

    assert len(augmented) == target_samples
    assert not np.isnan(augmented).any()
    assert not np.isinf(augmented).any()
    # Ensure audio was modified by augmentations
    assert not np.allclose(clean_audio, augmented)
