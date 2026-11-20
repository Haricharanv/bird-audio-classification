"""Input/Output helpers: serialized storage, atomic checkpointing, and disk space auditing."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import yaml


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON data from file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, path: Union[str, Path]) -> None:
    """Save data structure to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: Union[str, Path]) -> Any:
    """Load YAML data from file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_numpy(array: np.ndarray, path: Union[str, Path]) -> None:
    """Save numpy array to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(str(path))


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metric_val: float = 0.0,
    class_to_idx: Optional[Dict[str, int]] = None,
    cfg: Optional[Any] = None,
) -> None:
    """Atomically save training checkpoint along with species-to-index mapping and config.
    
    Args:
        path: Destination checkpoint path (.pt or .pth).
        model: PyTorch model instance.
        optimizer: Optimizer state dict if training.
        scheduler: Learning rate scheduler state.
        epoch: Epoch index reached.
        metric_val: Monitored validation metric score.
        class_to_idx: Dictionary mapping species labels to integer class indices.
        cfg: Configuration dictionary or Config instance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP / DataParallel if wrapped
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "epoch": epoch,
        "metric_val": metric_val,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "class_to_idx": class_to_idx,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
    }

    # Save to temp file first, then atomic rename to prevent corrupted checkpoints
    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint weights into model and optional optimizer/scheduler.
    
    Args:
        path: Path to checkpoint file.
        model: Target PyTorch model.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map tensors onto.
        
    Returns:
        Checkpoint dictionary containing metadata (epoch, class_to_idx, config, etc.).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def estimate_directory_size_gb(dir_path: Union[str, Path]) -> float:
    """Calculate total size of all files in directory in Gigabytes."""
    path = Path(dir_path)
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    return total_bytes / (1024.0**3)


def check_disk_space_or_abort(
    target_dir: Union[str, Path],
    incoming_gb: float,
    max_allowed_gb: float = 18.0,
) -> None:
    """Verify that writing incoming_gb will not violate disk caps (e.g. Kaggle 20GB limit).
    
    Args:
        target_dir: Target directory path.
        incoming_gb: Projected incoming size in Gigabytes.
        max_allowed_gb: Configurable maximum limit in Gigabytes.
        
    Raises:
        RuntimeError: If estimated total exceeds max_allowed_gb or available disk.
    """
    path = Path(target_dir)
    current_used_gb = estimate_directory_size_gb(path)
    projected_total = current_used_gb + incoming_gb

    # Check available system disk space
    stat_dir = path if path.exists() else path.parent
    if stat_dir.exists():
        total, used, free = shutil.disk_usage(stat_dir)
        free_gb = free / (1024.0**3)
        if incoming_gb > free_gb:
            raise RuntimeError(
                f"Aborting: Incoming data ({incoming_gb:.2f} GB) exceeds available disk space ({free_gb:.2f} GB)."
            )

    if projected_total > max_allowed_gb:
        raise RuntimeError(
            f"Aborting: Projected cache size ({projected_total:.2f} GB = {current_used_gb:.2f} GB existing + "
            f"{incoming_gb:.2f} GB incoming) exceeds configured limit of {max_allowed_gb:.2f} GB."
        )

