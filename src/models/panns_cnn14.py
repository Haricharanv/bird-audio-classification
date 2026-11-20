"""PANNs CNN14 architecture for audio pattern recognition (Kong et al., 2020)."""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.augment import SpecAugment
from src.features.melspec import build_frontend
from src.utils.logging import get_logger

logger = get_logger("bird_classification.panns")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class PANNsCNN14(nn.Module):
    """PANNs CNN14 model taking raw waveform and classifying bird vocalizations."""

    def __init__(self, cfg: Any, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.frontend = build_frontend(cfg)
        spec_cfg = cfg.augment.spec if hasattr(cfg, "augment") else {}
        self.spec_augment = SpecAugment(
            time_mask_num=int(getattr(spec_cfg, "time_mask_num", 2)),
            time_mask_max_len=int(getattr(spec_cfg, "time_mask_max_len", 40)),
            freq_mask_num=int(getattr(spec_cfg, "freq_mask_num", 2)),
            freq_mask_max_len=int(getattr(spec_cfg, "freq_mask_max_len", 24)),
            prob=float(getattr(spec_cfg, "spec_aug_prob", 0.5)),
        )

        dropout_p = float(getattr(cfg.model, "dropout", 0.2))

        # CNN14 feature extraction backbone
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc_final = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) and returning class logits."""
        spec = self.frontend(x)  # (B, 1, F, T)
        if self.training:
            spec = self.spec_augment(spec)

        x = self.conv_block1(spec, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(2, 2))
        x = self.conv_block5(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))

        # Global average and max pooling
        x_mean = torch.mean(x, dim=(2, 3))
        x_max, _ = torch.max(x.flatten(2), dim=2)
        feat = x_mean + x_max

        feat = F.relu_(self.fc1(self.dropout(feat)))
        logits = self.fc_final(self.dropout(feat))
        return logits

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
        head_params = list(self.fc1.parameters()) + list(self.fc_final.parameters())
        head_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in self.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def freeze_backbone(self) -> None:
        for block in [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6]:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

