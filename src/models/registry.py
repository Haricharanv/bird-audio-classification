"""Model registry and factory for audio classification backbones."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type
import torch.nn as nn

from src.models.ast_wrapper import ASTAudioClassifier
from src.models.baseline_cnn import BaselineCNN
from src.models.crnn import CRNN
from src.models.ensemble import EnsembleModel
from src.models.panns_cnn14 import PANNsCNN14
from src.models.pretrained_cnn import PretrainedCNN
from src.models.wav2vec2 import Wav2Vec2AudioClassifier
from src.utils.logging import get_logger

logger = get_logger("bird_classification.registry")

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline_cnn": BaselineCNN,
    "timm_cnn": PretrainedCNN,
    "crnn": CRNN,
    "ast": ASTAudioClassifier,
    "panns_cnn14": PANNsCNN14,
    "wav2vec2": Wav2Vec2AudioClassifier,
}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a custom model architecture."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def build_model(cfg: Any, num_classes: Optional[int] = None) -> nn.Module:
    """Build model instance specified by cfg.model.name taking raw waveform (B, T).
    
    Args:
        cfg: Configuration dictionary or Config instance.
        num_classes: Optional class count override (defaults to cfg.model.num_classes).
        
    Returns:
        Instantiated PyTorch nn.Module.
    """
    model_name = str(getattr(cfg.model, "name", "baseline_cnn")).lower()
    if num_classes is None:
        num_classes = int(getattr(cfg.model, "num_classes", 206))

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]
    logger.info(f"Instantiating model '{model_name}' (num_classes={num_classes})...")
    return model_cls(cfg=cfg, num_classes=num_classes)

