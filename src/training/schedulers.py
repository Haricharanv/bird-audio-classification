"""Optimizer and Learning Rate Scheduler factories with discriminative rates and warmup."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
    _LRScheduler,
)

from src.utils.logging import get_logger

logger = get_logger("bird_classification.schedulers")


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build AdamW optimizer with discriminative learning rates and decoupled weight decay.
    
    Args:
        model: PyTorch model.
        cfg: Configuration object.
        
    Returns:
        Instantiated AdamW optimizer.
    """
    opt_name = str(getattr(cfg.train, "optimizer", "adamw")).lower()
    weight_decay = float(getattr(cfg.train, "weight_decay", 0.01))
    head_lr = float(getattr(cfg.train, "head_lr", 1e-3))
    
    model_name = str(getattr(cfg.model, "name", "baseline_cnn")).lower()
    if model_name == "ast":
        backbone_lr = float(getattr(cfg.train, "ast_backbone_lr", 5e-5))
    else:
        backbone_lr = float(getattr(cfg.train, "backbone_lr", 2e-4))

    raw_model = model.module if hasattr(model, "module") else model

    # Check if model defines custom discriminative parameter groups
    if hasattr(raw_model, "get_param_groups"):
        raw_groups = raw_model.get_param_groups(head_lr=head_lr, backbone_lr=backbone_lr)
    else:
        raw_groups = [{"params": list(model.parameters()), "lr": head_lr}]

    # Filter out parameters with requires_grad=False and separate weight decay for biases/norms
    param_groups: List[Dict[str, Any]] = []
    for g in raw_groups:
        lr = g["lr"]
        decay_params = []
        no_decay_params = []

        for p in g["params"]:
            if not p.requires_grad:
                continue
            if p.ndim <= 1:
                no_decay_params.append(p)  # Biases, 1D norm scales
            else:
                decay_params.append(p)

        if decay_params:
            param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: '{opt_name}'. Choose 'adamw', 'adam', or 'sgd'.")

    logger.info(
        f"Built {opt_name.upper()} optimizer with {len(param_groups)} parameter groups "
        f"(head_lr={head_lr}, backbone_lr={backbone_lr}, weight_decay={weight_decay})."
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    epochs: int,
    steps_per_epoch: int,
) -> Tuple[Any, str]:
    """Build learning rate scheduler: OneCycleLR (step-wise) or CosineAnnealingLR (epoch/step-wise).
    
    Args:
        optimizer: PyTorch optimizer.
        cfg: Configuration object.
        epochs: Total number of training epochs.
        steps_per_epoch: Number of optimization steps per epoch.
        
    Returns:
        Tuple of (scheduler_instance, step_mode ['step' or 'epoch']).
    """
    sched_type = str(getattr(cfg.train, "scheduler", "one_cycle")).lower()
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_epochs = int(getattr(cfg.train, "warmup_epochs", 2))
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    if sched_type == "one_cycle":
        max_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=min(0.2, float(warmup_epochs / max(epochs, 1))),
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )
        logger.info(f"Built OneCycleLR scheduler for {total_steps} steps (step-wise).")
        return scheduler, "step"

    elif sched_type == "cosine":
        # Cosine with linear warmup
        if warmup_steps >= total_steps:
            warmup_steps = total_steps // 4
        cosine_steps = total_steps - warmup_steps

        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, cosine_steps), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
        logger.info(f"Built CosineAnnealingLR with {warmup_steps} warmup steps (step-wise).")
        return scheduler, "step"

    else:
        raise ValueError(f"Unsupported scheduler: '{sched_type}'. Choose 'one_cycle' or 'cosine'.")

