"""Audio classification models and registry package."""

from src.models.baseline_cnn import BaselineCNN
from src.models.pretrained_cnn import PretrainedCNN
from src.models.crnn import CRNN
from src.models.ast_wrapper import ASTAudioClassifier
from src.models.panns_cnn14 import PANNsCNN14
from src.models.wav2vec2 import Wav2Vec2AudioClassifier
from src.models.ensemble import EnsembleModel, load_ensemble_from_checkpoints
from src.models.registry import build_model, register_model, MODEL_REGISTRY

__all__ = [
    "BaselineCNN",
    "PretrainedCNN",
    "CRNN",
    "ASTAudioClassifier",
    "PANNsCNN14",
    "Wav2Vec2AudioClassifier",
    "EnsembleModel",
    "load_ensemble_from_checkpoints",
    "build_model",
    "register_model",
    "MODEL_REGISTRY",
]

