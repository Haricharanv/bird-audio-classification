"""Convolutional Recurrent Neural Network (CRNN) with BiGRU and Attention Pooling."""

from __future__ import annotations

from typing import Any, Dict, List
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.augment import SpecAugment
from src.features.melspec import build_frontend


class AttentionPooling(nn.Module):
    """Attention-weighted temporal pooling layer."""

    def __init__(self, in_features: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (B, T, C).
            
        Returns:
            Pooled tensor of shape (B, C).
        """
        # Attention scores: (B, T, 1)
        weights = self.attn(x)
        weights = F.softmax(weights, dim=1)
        # Weighted sum over time dimension
        pooled = torch.sum(x * weights, dim=1)  # (B, C)
        return pooled


class CRNN(nn.Module):
    """CRNN: timm features_only backbone -> collapse freq -> 2-layer BiGRU -> Attention Pooling -> FC."""

    def __init__(self, cfg: Any, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # 1. Frontend & SpecAugment
        self.frontend = build_frontend(cfg)
        spec_cfg = cfg.augment.spec if hasattr(cfg, "augment") else {}
        self.spec_augment = SpecAugment(
            time_mask_num=int(getattr(spec_cfg, "time_mask_num", 2)),
            time_mask_max_len=int(getattr(spec_cfg, "time_mask_max_len", 40)),
            freq_mask_num=int(getattr(spec_cfg, "freq_mask_num", 2)),
            freq_mask_max_len=int(getattr(spec_cfg, "freq_mask_max_len", 24)),
            prob=float(getattr(spec_cfg, "spec_aug_prob", 0.5)),
        )

        backbone_name = str(getattr(cfg.model, "backbone", "resnet34"))
        pretrained = bool(getattr(cfg.model, "pretrained", True))
        in_chans = int(getattr(cfg.model, "in_chans", 1))
        dropout_p = float(getattr(cfg.model, "dropout", 0.3))
        gru_hidden = int(getattr(cfg.model, "gru_hidden_size", 256))
        gru_layers = int(getattr(cfg.model, "gru_layers", 2))

        # 2. CNN Backbone extracting spatial feature maps
        self.cnn = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
        )
        # Determine output channel dimension of backbone
        feature_info = self.cnn.feature_info.channels()
        cnn_out_channels = feature_info[-1]

        # 3. Recurrent BiGRU layer
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if gru_layers > 1 else 0.0,
        )
        gru_out_features = gru_hidden * 2  # Bidirectional

        # 4. Attention Pooling over time
        self.attention_pool = AttentionPooling(gru_out_features)

        # 5. Classifier Head
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(gru_out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) and returning class logits."""
        spec = self.frontend(x)  # (B, 1, F, T)
        if self.training:
            spec = self.spec_augment(spec)

        # CNN feature extraction
        feats = self.cnn(spec)
        last_feat = feats[-1]  # (B, C, F_sub, T_sub)

        # Collapse frequency dimension by mean pooling
        feat_time = torch.mean(last_feat, dim=2)  # (B, C, T_sub)
        feat_time = feat_time.permute(0, 2, 1)    # (B, T_sub, C)

        # Recurrent processing
        gru_out, _ = self.gru(feat_time)          # (B, T_sub, 2*hidden)

        # Attention pooling over temporal sequence
        pooled = self.attention_pool(gru_out)     # (B, 2*hidden)

        dropped = self.dropout(pooled)
        logits = self.fc(dropped)                 # (B, num_classes)
        return logits

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
        """Separate CNN backbone from GRU and classifier head."""
        backbone_params = list(self.cnn.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def freeze_backbone(self) -> None:
        """Freeze CNN backbone weights."""
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze CNN backbone weights."""
        for param in self.parameters():
            param.requires_grad = True

