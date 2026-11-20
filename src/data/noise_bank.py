"""Background noise bank extracted from unlabeled soundscapes for realistic audio augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import soundfile as sf
import librosa

from src.utils.logging import get_logger

logger = get_logger("bird_classification.noise_bank")


def _generate_synthetic_ambient_noise(num_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate naturalistic ambient background noise (pink/brown noise blend) as a fallback."""
    # Generate white noise
    white = rng.randn(num_samples)
    # Apply 1/f spectral filtering (pink noise approximation via FFT)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = 1e-4  # avoid div by zero
    spectral_shaping = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(fft_white * spectral_shaping, n=num_samples)
    
    # Scale amplitude
    peak = np.max(np.abs(pink)) + 1e-8
    pink = (pink / peak) * 0.15
    return pink.astype(np.float32)


def build_noise_bank(cfg: Any) -> np.ndarray:
    """Extract random 5-second background crops from train_soundscapes and save to disk.
    
    Args:
        cfg: Configuration object.
        
    Returns:
        Numpy array of shape (noise_bank_size, target_samples) containing noise clips.
    """
    raw_dir = Path(cfg.data.raw_dir)
    soundscapes_dir = raw_dir / "train_soundscapes"
    out_path = Path(cfg.data.noise_bank_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_sr = int(getattr(cfg.audio, "sample_rate", 32000))
    duration_sec = float(getattr(cfg.audio, "duration_sec", 5.0))
    target_samples = int(duration_sec * target_sr)
    bank_size = int(getattr(cfg.data, "noise_bank_size", 300))
    seed = int(getattr(cfg, "seed", 42))

    rng = np.random.RandomState(seed)
    soundscape_files = list(soundscapes_dir.glob("*.ogg")) if soundscapes_dir.is_dir() else []
    
    noise_crops = []

    if soundscape_files:
        logger.info(f"Extracting {bank_size} noise segments from {len(soundscape_files)} soundscape files...")
        file_pool = list(soundscape_files)
        rng.shuffle(file_pool)

        file_idx = 0
        while len(noise_crops) < bank_size and file_idx < len(file_pool) * 5:
            sc_file = file_pool[file_idx % len(file_pool)]
            file_idx += 1
            try:
                # Read soundscape file
                y, sr = sf.read(str(sc_file), dtype="float32", always_2d=True)
                if y.shape[1] > 1:
                    y = np.mean(y, axis=1)
                else:
                    y = y[:, 0]
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                
                if len(y) >= target_samples:
                    # Random 5-second crop
                    max_start = len(y) - target_samples
                    start = rng.randint(0, max_start + 1)
                    crop = y[start : start + target_samples]
                    
                    # Normalize peak
                    peak = np.max(np.abs(crop))
                    if peak > 1e-4:
                        crop = crop / peak
                        noise_crops.append(crop.astype(np.float32))
            except Exception as e:
                logger.debug(f"Failed to read soundscape {sc_file}: {e}")

    # Fallback to synthesized ambient background if insufficient soundscapes exist
    if len(noise_crops) < bank_size:
        needed = bank_size - len(noise_crops)
        logger.info(f"Generating {needed} synthetic ambient noise segments as soundscape supplement...")
        for _ in range(needed):
            synth_crop = _generate_synthetic_ambient_noise(target_samples, rng)
            noise_crops.append(synth_crop)

    noise_bank = np.stack(noise_crops[:bank_size], axis=0).astype(np.float32)
    np.save(str(out_path), noise_bank)
    logger.info(f"Saved noise bank with shape {noise_bank.shape} ({noise_bank.nbytes / 1e6:.2f} MB) to {out_path}.")
    return noise_bank


def load_noise_bank(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load noise bank from file if present."""
    p = Path(path)
    if p.is_file():
        return np.load(str(p))
    return None

