"""Loss functions for bird classification: Cross-Entropy, Weighted CE, Focal Loss, and BCE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger("bird_classification.losses")


class CrossEntropyWithSmoothing(nn.Module):
    """Cross Entropy supporting both 1D hard labels and 2D soft/Mixup distributions with label smoothing."""

    def __init__(self, label_smoothing: float = 0.05, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Predicted unnormalized logits (B, num_classes).
            targets: Integer indices (B,) or one-hot / mixed distributions (B, num_classes).
        """
        num_classes = logits.size(-1)

        # 1. Targets are 1D integer class indices
        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(-1) == 1):
            targets_1d = targets.view(-1).long()
            return F.cross_entropy(
                logits,
                targets_1d,
                weight=self.weight,
                label_smoothing=self.label_smoothing,
            )

        # 2. Targets are 2D soft/one-hot probabilities (e.g. from Mixup/CutMix)
        targets_2d = targets.float()
        if self.label_smoothing > 0.0:
            targets_2d = (1.0 - self.label_smoothing) * targets_2d + (self.label_smoothing / num_classes)

        log_probs = F.log_softmax(logits, dim=-1)
        if self.weight is not None:
            loss = -torch.sum(targets_2d * log_probs * self.weight.unsqueeze(0), dim=-1)
        else:
            loss = -torch.sum(targets_2d * log_probs, dim=-1)

        return loss.mean()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with focusing parameter gamma and optional alpha class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            targets_onehot = F.one_hot(targets.long(), num_classes=num_classes).float()
        else:
            targets_onehot = targets.float()

        if self.label_smoothing > 0.0:
            targets_onehot = (1.0 - self.label_smoothing) * targets_onehot + (self.label_smoothing / num_classes)

        focal_weight = torch.pow(1.0 - probs, self.gamma)
        loss = -focal_weight * log_probs * targets_onehot

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                loss = loss * self.alpha.unsqueeze(0)
            else:
                loss = loss * self.alpha

        return loss.sum(dim=-1).mean()


def build_loss(
    cfg: Any,
    class_counts: Optional[Dict[str, int]] = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Build loss function based on configuration.
    
    Args:
        cfg: Configuration object.
        class_counts: Optional dictionary of species counts for weighted CE.
        device: Device to place weights onto.
        
    Returns:
        Instantiated loss Module.
    """
    loss_type = str(getattr(cfg.train, "loss", "cross_entropy")).lower()
    label_smoothing = float(getattr(cfg.train, "label_smoothing", 0.05))
    focal_gamma = float(getattr(cfg.train, "focal_gamma", 2.0))
    is_multilabel = str(getattr(cfg, "task", "multiclass")).lower() == "multilabel"

    if is_multilabel or loss_type == "bce":
        logger.info("Using BCEWithLogitsLoss for multi-label classification.")
        return nn.BCEWithLogitsLoss()

    if loss_type == "weighted_ce":
        if class_counts is not None and len(class_counts) > 0:
            # Inverse square root weighting: w_c = 1 / sqrt(N_c)
            counts_arr = np.array(list(class_counts.values()), dtype=np.float32)
            inv_sqrt = 1.0 / np.sqrt(np.maximum(counts_arr, 1.0))
            weights = inv_sqrt / inv_sqrt.sum() * len(counts_arr)  # normalized so mean is 1.0
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            logger.info(f"Using Weighted Cross Entropy with inverse sqrt class weights (num_classes={len(weights)}).")
            return CrossEntropyWithSmoothing(label_smoothing=label_smoothing, weight=weight_tensor)
        else:
            logger.warning("Class counts not provided for weighted_ce; defaulting to standard CE.")
            return CrossEntropyWithSmoothing(label_smoothing=label_smoothing)

    elif loss_type == "focal":
        logger.info(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing}).")
        return FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)

    # Default: cross_entropy
    logger.info(f"Using CrossEntropyLossWithLabelSmoothing (smoothing={label_smoothing}).")
    return CrossEntropyWithSmoothing(label_smoothing=label_smoothing)

