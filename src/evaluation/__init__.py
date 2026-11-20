"""Evaluation, metrics, analysis, and interpretability package."""

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_per_collection_metrics,
)
from src.evaluation.analysis import (
    plot_and_save_confusion_matrix,
    find_top_confused_pairs,
    save_confident_misclassifications,
    generate_full_evaluation_report,
)
from src.evaluation.gradcam import GradCAM

__all__ = [
    "compute_classification_metrics",
    "compute_per_collection_metrics",
    "plot_and_save_confusion_matrix",
    "find_top_confused_pairs",
    "save_confident_misclassifications",
    "generate_full_evaluation_report",
    "GradCAM",
]

