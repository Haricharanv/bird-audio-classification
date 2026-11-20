"""PyTorch nn.Module audio frontends for on-GPU feature extraction.

Computes Log-Mel, MFCC, STFT, and PCEN representations with per-sample standardization.
Operates on raw waveform tensors (B, T) or (B, 1, T) and outputs (B, 1, F, T').
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torchaudio.transforms as T


class AudioFrontend(nn.Module):
    """Base class for differentiable audio frontend feature extractors."""

    def __init__(self, standardize: bool = True):
        super().__init__()
        self.standardize = standardize

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        """Format input waveform to (B, T) float32."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() > 2:
            raise ValueError(f"Expected waveform with shape (B, T) or (B, 1, T), got {x.shape}")
        return x.float()

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample z-score standardization over frequency and time axes."""
        if not self.standardize:
            return x
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-6)


class MelSpectrogramFrontend(AudioFrontend):
    """GPU-compatible Log-Mel Spectrogram extraction module."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        power: float = 2.0,
        top_db: float = 80.0,
        standardize: bool = True,
    ):
        super().__init__(standardize=standardize)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=top_db)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Log-Mel spectrogram.
        
        Args:
            x: Raw audio waveform (B, T) or (B, 1, T).
            
        Returns:
            Normalized spectrogram tensor (B, 1, n_mels, time_steps).
        """
        x = self._format_input(x)
        mel = self.mel_transform(x)          # (B, n_mels, time_steps)
        log_mel = self.amp_to_db(mel)        # (B, n_mels, time_steps)
        log_mel = log_mel.unsqueeze(1)       # (B, 1, n_mels, time_steps)
        return self._standardize(log_mel)


class MFCCFrontend(AudioFrontend):
    """GPU-compatible Mel-Frequency Cepstral Coefficients (MFCC) extraction module."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        standardize: bool = True,
    ):
        super().__init__(standardize=standardize)
        melkwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "f_min": fmin,
            "f_max": fmax,
            "n_mels": n_mels,
            "power": 2.0,
            "center": True,
            "pad_mode": "reflect",
            "norm": "slaney",
            "mel_scale": "slaney",
        }
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=melkwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract MFCCs.
        
        Args:
            x: Raw audio waveform (B, T).
            
        Returns:
            Normalized MFCC tensor (B, 1, n_mfcc, time_steps).
        """
        x = self._format_input(x)
        mfcc = self.mfcc_transform(x)        # (B, n_mfcc, time_steps)
        mfcc = mfcc.unsqueeze(1)             # (B, 1, n_mfcc, time_steps)
        return self._standardize(mfcc)


class STFTFrontend(AudioFrontend):
    """GPU-compatible Log Magnitude STFT spectrogram extraction module."""

    def __init__(
        self,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        top_db: float = 80.0,
        standardize: bool = True,
    ):
        super().__init__(standardize=standardize)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.top_db = top_db
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Log Magnitude STFT.
        
        Args:
            x: Raw audio waveform (B, T).
            
        Returns:
            Normalized Log-STFT tensor (B, 1, n_fft // 2 + 1, time_steps).
        """
        x = self._format_input(x)
        stft_c = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mag = torch.abs(stft_c).clamp(min=1e-8)
        log_stft = 20.0 * torch.log10(mag)
        
        # Clamp dynamic range to top_db below maximum
        max_val = log_stft.amax(dim=(-2, -1), keepdim=True)
        log_stft = torch.maximum(log_stft, max_val - self.top_db)
        
        log_stft = log_stft.unsqueeze(1)     # (B, 1, freq_bins, time_steps)
        return self._standardize(log_stft)


class PCENFrontend(AudioFrontend):
    """Per-Channel Energy Normalization (PCEN) frontend module.
    
    PCEN replaces logarithmic compression with adaptive gain control,
    enhancing foreground bird vocalisations relative to stationary environmental noise.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        s: float = 0.025,
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
        eps: float = 1e-6,
        standardize: bool = True,
    ):
        super().__init__(standardize=standardize)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract PCEN spectrogram.
        
        Args:
            x: Raw audio waveform (B, T).
            
        Returns:
            Normalized PCEN tensor (B, 1, n_mels, time_steps).
        """
        x = self._format_input(x)
        # Power mel spectrogram: (B, n_mels, T)
        E = self.mel_transform(x)
        
        # Causal IIR smoother along time axis
        batch_size, n_mels, num_frames = E.shape
        M = torch.empty_like(E)
        last_m = E[..., 0]
        M[..., 0] = last_m
        for t in range(1, num_frames):
            last_m = (1.0 - self.s) * last_m + self.s * E[..., t]
            M[..., t] = last_m
        
        # Adaptive AGC: (E / (eps + M)^alpha + delta)^r - delta^r
        smooth = torch.pow(self.eps + M, self.alpha)
        pcen = torch.pow(E / smooth + self.delta, self.r) - (self.delta**self.r)
        
        pcen = pcen.unsqueeze(1)             # (B, 1, n_mels, time_steps)
        return self._standardize(pcen)


def build_frontend(cfg: Any) -> AudioFrontend:
    """Build the configured audio frontend module.
    
    Args:
        cfg: Configuration containing audio parameters.
        
    Returns:
        Instantiated AudioFrontend subclass.
    """
    audio_cfg = cfg.audio if hasattr(cfg, "audio") else cfg.get("audio", {})
    feat_type = (
        audio_cfg.feature_type
        if hasattr(audio_cfg, "feature_type")
        else audio_cfg.get("feature_type", "logmel")
    ).lower()

    sr = getattr(audio_cfg, "sample_rate", 32000)
    n_fft = getattr(audio_cfg, "n_fft", 1024)
    win_length = getattr(audio_cfg, "win_length", 1024)
    hop_length = getattr(audio_cfg, "hop_length", 320)
    n_mels = getattr(audio_cfg, "n_mels", 128)
    fmin = getattr(audio_cfg, "fmin", 20.0)
    fmax = getattr(audio_cfg, "fmax", 16000.0)
    power = getattr(audio_cfg, "power", 2.0)
    top_db = getattr(audio_cfg, "top_db", 80.0)
    standardize = getattr(audio_cfg, "standardize", True)

    if feat_type == "logmel":
        return MelSpectrogramFrontend(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=power,
            top_db=top_db,
            standardize=standardize,
        )
    elif feat_type == "mfcc":
        n_mfcc = getattr(audio_cfg, "n_mfcc", 40)
        return MFCCFrontend(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            standardize=standardize,
        )
    elif feat_type == "stft":
        return STFTFrontend(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            top_db=top_db,
            standardize=standardize,
        )
    elif feat_type == "pcen":
        return PCENFrontend(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            standardize=standardize,
        )
    else:
        raise ValueError(
            f"Unsupported feature_type: '{feat_type}'. Supported: 'logmel', 'mfcc', 'stft', 'pcen'."
        )

