"""Unit tests for on-GPU audio frontend feature extraction and tensor shapes."""

import pytest
import torch

from src.features.melspec import (
    MelSpectrogramFrontend,
    MFCCFrontend,
    STFTFrontend,
    PCENFrontend,
    build_frontend,
)
from src.utils.config import Config


@pytest.fixture
def dummy_audio_batch():
    """Create a batch of 5-second audio waveforms at 32 kHz."""
    batch_size = 2
    duration_sec = 5.0
    sr = 32000
    num_samples = int(duration_sec * sr)  # 160,000 samples
    return torch.randn(batch_size, num_samples)


def test_logmel_shape_and_standardization(dummy_audio_batch):
    frontend = MelSpectrogramFrontend(
        sample_rate=32000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=128,
        standardize=True,
    )
    out = frontend(dummy_audio_batch)
    
    # Expected: (batch_size, 1, n_mels=128, frames=501)
    assert out.dim() == 4
    assert out.shape[0] == dummy_audio_batch.shape[0]
    assert out.shape[1] == 1
    assert out.shape[2] == 128
    assert out.shape[3] == 501
    
    # Check standardization per sample
    mean = out.mean(dim=(-2, -1))
    std = out.std(dim=(-2, -1))
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)


def test_mfcc_shape(dummy_audio_batch):
    frontend = MFCCFrontend(
        sample_rate=32000,
        n_mfcc=40,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=128,
        standardize=True,
    )
    out = frontend(dummy_audio_batch)
    assert out.shape == (dummy_audio_batch.shape[0], 1, 40, 501)


def test_stft_shape(dummy_audio_batch):
    frontend = STFTFrontend(
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        standardize=True,
    )
    out = frontend(dummy_audio_batch)
    # n_fft // 2 + 1 = 513
    assert out.shape == (dummy_audio_batch.shape[0], 1, 513, 501)


def test_pcen_shape(dummy_audio_batch):
    frontend = PCENFrontend(
        sample_rate=32000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=128,
        standardize=True,
    )
    out = frontend(dummy_audio_batch)
    assert out.shape == (dummy_audio_batch.shape[0], 1, 128, 501)


def test_single_audio_1d_input():
    frontend = MelSpectrogramFrontend(sample_rate=32000, n_mels=128)
    waveform_1d = torch.randn(160000)
    out = frontend(waveform_1d)
    assert out.shape == (1, 1, 128, 501)


def test_build_frontend_from_config():
    cfg = Config({
        "audio": {
            "feature_type": "logmel",
            "sample_rate": 32000,
            "n_fft": 1024,
            "hop_length": 320,
            "win_length": 1024,
            "n_mels": 128,
            "fmin": 20.0,
            "fmax": 16000.0,
            "standardize": True,
        }
    })
    frontend = build_frontend(cfg)
    assert isinstance(frontend, MelSpectrogramFrontend)
    x = torch.randn(1, 160000)
    out = frontend(x)
    assert out.shape == (1, 1, 128, 501)

