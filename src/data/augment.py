"""Audio augmentation modules: waveform transforms, SpecAugment, and batch Mixup/CutMix."""

from __future__ import annotations

import random
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import librosa

from src.utils.logging import get_logger

logger = get_logger("bird_classification.augment")


class WaveformAugmenter:
    """Waveform-level augmentation pipeline applying noise, gain, pitch, stretch, and shift."""

    def __init__(
        self,
        cfg: Any,
        noise_bank: Optional[np.ndarray] = None,
    ):
        wf_cfg = cfg.augment.waveform if hasattr(cfg, "augment") else {}
        self.noise_bank = noise_bank
        self.sample_rate = int(getattr(cfg.audio, "sample_rate", 32000))

        # Augmentation probabilities and hyperparameters
        self.noise_prob = float(getattr(wf_cfg, "noise_bank_prob", 0.5))
        self.snr_min = float(getattr(wf_cfg, "noise_snr_min_db", 3.0))
        self.snr_max = float(getattr(wf_cfg, "noise_snr_max_db", 20.0))

        self.gain_prob = float(getattr(wf_cfg, "gain_prob", 0.5))
        self.gain_min = float(getattr(wf_cfg, "gain_min_db", -6.0))
        self.gain_max = float(getattr(wf_cfg, "gain_max_db", 6.0))

        self.pitch_prob = float(getattr(wf_cfg, "pitch_shift_prob", 0.2))
        self.pitch_semitones = int(getattr(wf_cfg, "pitch_shift_semitones", 1))

        self.stretch_prob = float(getattr(wf_cfg, "time_stretch_prob", 0.2))
        self.stretch_min = float(getattr(wf_cfg, "time_stretch_min", 0.9))
        self.stretch_max = float(getattr(wf_cfg, "time_stretch_max", 1.1))

        self.shift_prob = float(getattr(wf_cfg, "time_shift_prob", 0.5))

    def _add_noise(self, x: np.ndarray) -> np.ndarray:
        """Mix in realistic background noise from the soundscape noise bank."""
        if self.noise_bank is None or len(self.noise_bank) == 0:
            return x
        
        # Pick random noise segment
        idx = random.randint(0, len(self.noise_bank) - 1)
        noise = self.noise_bank[idx]

        # Match lengths if necessary
        if len(noise) < len(x):
            repeats = int(np.ceil(len(x) / len(noise)))
            noise = np.tile(noise, repeats)[: len(x)]
        elif len(noise) > len(x):
            start = random.randint(0, len(noise) - len(x))
            noise = noise[start : start + len(x)]

        # Calculate RMS of signal and noise
        sig_rms = np.sqrt(np.mean(np.square(x)))
        if sig_rms < 1e-6:
            return x
        
        noise_rms = np.sqrt(np.mean(np.square(noise)))
        if noise_rms < 1e-6:
            return x

        # Desired SNR in dB
        snr_db = random.uniform(self.snr_min, self.snr_max)
        snr_linear = 10.0 ** (snr_db / 20.0)
        scale = sig_rms / (snr_linear * noise_rms)

        mixed = x + scale * noise
        # Normalize peak
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
        return mixed

    def _apply_gain(self, x: np.ndarray) -> np.ndarray:
        """Apply random gain in dB."""
        gain_db = random.uniform(self.gain_min, self.gain_max)
        gain_factor = 10.0 ** (gain_db / 20.0)
        scaled = x * gain_factor
        return np.clip(scaled, -1.0, 1.0)

    def _pitch_shift(self, x: np.ndarray) -> np.ndarray:
        """Apply mild pitch shift (+- semitones)."""
        steps = random.choice([-self.pitch_semitones, self.pitch_semitones])
        try:
            return librosa.effects.pitch_shift(x, sr=self.sample_rate, n_steps=steps)
        except Exception:
            return x

    def _time_stretch(self, x: np.ndarray) -> np.ndarray:
        """Apply mild time stretch (0.9 - 1.1x) with fixed length preservation."""
        rate = random.uniform(self.stretch_min, self.stretch_max)
        target_len = len(x)
        try:
            stretched = librosa.effects.time_stretch(x, rate=rate)
            if len(stretched) < target_len:
                # Pad
                padded = np.zeros(target_len, dtype=x.dtype)
                padded[: len(stretched)] = stretched
                return padded
            else:
                # Crop
                return stretched[:target_len]
        except Exception:
            return x

    def _time_shift(self, x: np.ndarray) -> np.ndarray:
        """Apply circular time shift / roll."""
        shift = random.randint(-len(x) // 2, len(x) // 2)
        return np.roll(x, shift)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply configured waveform augmentations in sequence.
        
        Args:
            waveform: 1D numpy array.
            
        Returns:
            Augmented 1D numpy array.
        """
        x = waveform.copy()

        if random.random() < self.noise_prob:
            x = self._add_noise(x)

        if random.random() < self.gain_prob:
            x = self._apply_gain(x)

        if random.random() < self.pitch_prob:
            x = self._pitch_shift(x)

        if random.random() < self.stretch_prob:
            x = self._time_stretch(x)

        if random.random() < self.shift_prob:
            x = self._time_shift(x)

        return x.astype(np.float32)


class SpecAugment(nn.Module):
    """SpecAugment on spectrogram representations (Time & Frequency Masking)."""

    def __init__(
        self,
        time_mask_num: int = 2,
        time_mask_max_len: int = 40,
        freq_mask_num: int = 2,
        freq_mask_max_len: int = 24,
        prob: float = 0.5,
    ):
        super().__init__()
        self.time_mask_num = time_mask_num
        self.time_mask_max_len = time_mask_max_len
        self.freq_mask_num = freq_mask_num
        self.freq_mask_max_len = freq_mask_max_len
        self.prob = prob

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time and frequency masks to spectrogram tensor (B, C, F, T)."""
        if not self.training or random.random() > self.prob:
            return spec

        cloned = spec.clone()
        B, C, F, T = cloned.shape

        # Frequency masking
        for _ in range(self.freq_mask_num):
            f_len = random.randint(0, min(self.freq_mask_max_len, F))
            f_start = random.randint(0, F - f_len)
            cloned[:, :, f_start : f_start + f_len, :] = 0.0

        # Time masking
        for _ in range(self.time_mask_num):
            t_len = random.randint(0, min(self.time_mask_max_len, T))
            t_start = random.randint(0, T - t_len)
            cloned[:, :, :, t_start : t_start + t_len] = 0.0

        return cloned


class BatchMixupCutmix:
    """Batch-level Mixup and CutMix data augmentation."""

    def __init__(
        self,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        cutmix_alpha: float = 0.4,
        cutmix_prob: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup or CutMix to batch tensors.
        
        Args:
            x: Input batch (B, ...).
            y: Target tensor (B,) integer indices or (B, num_classes) multi-hot floats.
            num_classes: Total number of species classes.
            
        Returns:
            Tuple of (augmented_x, mixed_targets).
        """
        batch_size = x.size(0)
        if batch_size <= 1:
            if y.dim() == 1:
                y = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
            return x, y

        # Convert 1D integer targets to float one-hot
        if y.dim() == 1:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        else:
            y_onehot = y.float()

        # Decide whether to apply CutMix or Mixup
        use_cutmix = random.random() < self.cutmix_prob and x.dim() == 4
        use_mixup = (not use_cutmix) and (random.random() < self.mixup_prob)

        if not use_cutmix and not use_mixup:
            return x, y_onehot

        perm = torch.randperm(batch_size, device=x.device)

        if use_cutmix:
            # Sample bounding box
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            B, C, H, W = x.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            x_aug = x.clone()
            x_aug[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
            
            # Adjusted lambda by box area
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
            y_aug = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
            return x_aug, y_aug

        # Standard Mixup
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        x_aug = lam * x + (1.0 - lam) * x[perm]
        y_aug = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
        return x_aug, y_aug

