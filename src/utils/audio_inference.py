"""Shared sliding-window slicing and confidence/energy-weighted prediction pooling."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


def slice_sliding_windows(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    window_sec: float = 5.0,
    hop_sec: float = 2.5,
) -> Tuple[np.ndarray, List[float]]:
    """Split a 1D waveform into fixed-duration overlapping windows and timestamps.

    Args:
        waveform: 1D audio array.
        sample_rate: Sampling rate in Hz.
        window_sec: Window duration in seconds.
        hop_sec: Hop step in seconds.

    Returns:
        Tuple of (windows array of shape [W, window_samples], list of start timestamps in sec).
    """
    window_samples = int(window_sec * sample_rate)
    hop_samples = max(1, int(hop_sec * sample_rate))
    total_len = len(waveform)

    if total_len <= window_samples:
        if total_len == 0:
            return np.zeros((1, window_samples), dtype=np.float32), [0.0]
        repeats = int(np.ceil(window_samples / total_len))
        padded = np.tile(waveform, repeats)[:window_samples].astype(np.float32)
        return padded[np.newaxis, :], [0.0]

    windows: List[np.ndarray] = []
    timestamps: List[float] = []
    for start in range(0, total_len - window_samples + 1, hop_samples):
        windows.append(waveform[start : start + window_samples])
        timestamps.append(start / float(sample_rate))

    tail_start = total_len - window_samples
    if tail_start > 0 and (total_len - window_samples) % hop_samples != 0:
        windows.append(waveform[tail_start:total_len])
        timestamps.append(tail_start / float(sample_rate))

    return np.stack(windows, axis=0).astype(np.float32), timestamps


def aggregate_window_probs(
    probs: np.ndarray,
    windows: np.ndarray | None = None,
    mode: str = "confidence_energy",
) -> np.ndarray:
    """Aggregate per-window class probabilities into a single recording-level vector.

    Weighting active vocalization windows higher avoids diluting bird calls with
    silent or ambient background segments in longer recordings.

    Args:
        probs: Array of shape (num_windows, num_classes).
        windows: Optional waveform windows array (num_windows, window_samples).
        mode: 'confidence_energy' | 'energy' | 'mean' | 'max'.

    Returns:
        1D probability array of shape (num_classes,).
    """
    if probs.ndim != 2 or len(probs) == 0:
        raise ValueError(f"Expected 2D non-empty probs array, got shape {probs.shape}")
    if len(probs) == 1 or mode == "mean":
        return np.mean(probs, axis=0)
    if mode == "max":
        raw = np.max(probs, axis=0)
        denom = float(np.sum(raw))
        return raw / denom if denom > 1e-8 else np.mean(probs, axis=0)

    if windows is not None and len(windows) == len(probs):
        rms = np.sqrt(np.mean(np.square(windows), axis=1)) + 1e-6
    else:
        rms = np.ones(len(probs), dtype=np.float32)

    if mode == "energy":
        weights = rms
    else:  # "confidence_energy"
        max_conf = np.max(probs, axis=1)
        weights = rms * np.square(max_conf)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-8:
        return np.mean(probs, axis=0)

    norm_weights = (weights / weight_sum)[:, np.newaxis]
    return np.sum(probs * norm_weights, axis=0)
