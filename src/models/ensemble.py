"""Ensemble model averaging class probabilities from multiple checkpointed models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.io import load_checkpoint
from src.utils.logging import get_logger

logger = get_logger("bird_classification.ensemble")


class EnsembleModel(nn.Module):
    """Ensemble model averaging softmax (or sigmoid) probabilities across multiple architectures."""

    def __init__(self, models: List[nn.Module], is_multilabel: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.is_multilabel = is_multilabel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) or batched sliding windows (B, W, T).
        
        Returns:
            Ensemble average probabilities (B, num_classes).
        """
        all_probs: List[torch.Tensor] = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                if self.is_multilabel:
                    probs = torch.sigmoid(logits)
                else:
                    probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        # Average probabilities across ensemble members
        stacked = torch.stack(all_probs, dim=0)  # (M, B, num_classes)
        avg_probs = torch.mean(stacked, dim=0)   # (B, num_classes)
        return avg_probs


def load_ensemble_from_checkpoints(
    checkpoint_paths: List[Union[str, Path]],
    cfg: Any,
    device: Union[str, torch.device] = "cpu",
) -> EnsembleModel:
    """Load models from checkpoints and instantiate EnsembleModel.
    
    Args:
        checkpoint_paths: List of checkpoint file paths (.pt).
        cfg: Configuration object.
        device: Device to place models on.
        
    Returns:
        Instantiated and weighted EnsembleModel.
    """
    from src.models.registry import build_model

    models = []
    is_multilabel = str(getattr(cfg, "task", "multiclass")).lower() == "multilabel"

    for cp_path in checkpoint_paths:
        path = Path(cp_path)
        logger.info(f"Loading checkpoint for ensemble: {path.name}...")
        ckpt = torch.load(path, map_location=device)
        model_cfg = ckpt.get("config", cfg)
        num_classes = len(ckpt.get("class_to_idx", {})) or int(getattr(cfg.model, "num_classes", 206))
        
        model = build_model(model_cfg, num_classes=num_classes)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    logger.info(f"Loaded {len(models)} models into Ensemble.")
    return EnsembleModel(models, is_multilabel=is_multilabel)

