"""Comprehensive bioacoustic evaluation metrics: Macro-F1, Top-k, BirdCLEF ROC-AUC, and cmAP."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from src.utils.logging import get_logger

logger = get_logger("bird_classification.metrics")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    is_multilabel: bool = False,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: 1D array of class indices (N,) or 2D binary indicator (N, C) for multilabel.
        y_pred_probs: 2D array of predicted probabilities (N, C).
        is_multilabel: If True, uses multilabel evaluation logic.
        class_names: Optional list of class string names of length C.
        
    Returns:
        Dictionary of scalar metrics and per-class metrics DataFrame.
    """
    N, num_classes = y_pred_probs.shape
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    results: Dict[str, Any] = {}

    if is_multilabel:
        # Multilabel evaluation
        binary_preds = (y_pred_probs >= 0.5).astype(int)
        macro_f1 = float(f1_score(y_true, binary_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, binary_preds, average="weighted", zero_division=0))
        top1_acc = float(np.mean(binary_preds == y_true))
        top3_acc = top1_acc

        # Class mean Average Precision (cmAP)
        ap_scores = []
        for c in range(num_classes):
            if np.sum(y_true[:, c]) > 0:
                ap = average_precision_score(y_true[:, c], y_pred_probs[:, c])
                ap_scores.append(ap)
        cmap = float(np.mean(ap_scores)) if ap_scores else 0.0

        results["macro_f1"] = macro_f1
        results["weighted_f1"] = weighted_f1
        results["top1_accuracy"] = top1_acc
        results["top3_accuracy"] = top3_acc
        results["cmap"] = cmap
        results["roc_auc"] = cmap

        # Per-class metrics
        prec, rec, f1, supp = precision_recall_fscore_support(
            y_true, binary_preds, average=None, zero_division=0
        )
        per_class_df = pd.DataFrame({
            "species": class_names,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supp,
        })
        results["per_class_report"] = per_class_df
        return results

    # Single-label multiclass evaluation
    pred_labels = np.argmax(y_pred_probs, axis=1)

    macro_f1 = float(f1_score(y_true, pred_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, pred_labels, average="weighted", zero_division=0))
    top1_acc = float(np.mean(pred_labels == y_true))

    top3_k = min(3, num_classes)
    top3_acc = float(top_k_accuracy_score(
        y_true, y_pred_probs, k=top3_k, labels=list(range(num_classes))
    ))

    # BirdCLEF-style macro ROC-AUC: average over classes with at least one positive and one negative
    roc_auc_per_class: List[float] = []
    ap_per_class: List[float] = []

    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        pos_count = np.sum(y_true_c)
        if 0 < pos_count < N:
            try:
                score = roc_auc_score(y_true_c, y_pred_probs[:, c])
                roc_auc_per_class.append(score)
            except ValueError:
                pass
            ap = average_precision_score(y_true_c, y_pred_probs[:, c])
            ap_per_class.append(ap)
        else:
            ap_per_class.append(0.0)

    macro_roc_auc = float(np.mean(roc_auc_per_class)) if roc_auc_per_class else 0.0

    # Per-class precision, recall, F1, support
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, pred_labels, labels=list(range(num_classes)), average=None, zero_division=0
    )
    per_class_df = pd.DataFrame({
        "species": class_names,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": supp,
        "average_precision": ap_per_class,
    })

    results["macro_f1"] = macro_f1
    results["weighted_f1"] = weighted_f1
    results["top1_accuracy"] = top1_acc
    results["top3_accuracy"] = top3_acc
    results["roc_auc"] = macro_roc_auc
    results["per_class_report"] = per_class_df

    return results


def compute_per_collection_metrics(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    is_multilabel: bool = False,
) -> pd.DataFrame:
    """Calculate metrics broken down by audio collection source (XC, iNat, CSA).
    
    Args:
        df_meta: DataFrame with 'collection' column.
        y_true: True targets.
        y_pred_probs: Predicted probabilities.
        is_multilabel: Multilabel task flag.
        
    Returns:
        DataFrame with per-collection metric comparison.
    """
    collections = df_meta["collection"].unique()
    rows = []

    for coll in collections:
        mask = (df_meta["collection"] == coll).to_numpy()
        sub_n = np.sum(mask)
        if sub_n == 0:
            continue

        sub_true = y_true[mask]
        sub_probs = y_pred_probs[mask]

        if is_multilabel:
            bin_preds = (sub_probs >= 0.5).astype(int)
            f1 = float(f1_score(sub_true, bin_preds, average="macro", zero_division=0))
            top1 = float(np.mean(bin_preds == sub_true))
            top3 = top1
            auc = 0.0
        else:
            sub_preds = np.argmax(sub_probs, axis=1)
            f1 = float(f1_score(sub_true, sub_preds, average="macro", zero_division=0))
            top1 = float(np.mean(sub_preds == sub_true))
            top3_k = min(3, sub_probs.shape[1])
            top3 = float(top_k_accuracy_score(
                sub_true, sub_probs, k=top3_k, labels=list(range(sub_probs.shape[1]))
            )) if len(np.unique(sub_true)) > 1 else top1

            auc_scores = []
            for c in range(sub_probs.shape[1]):
                y_c = (sub_true == c).astype(int)
                if 0 < np.sum(y_c) < len(y_c):
                    try:
                        auc_scores.append(roc_auc_score(y_c, sub_probs[:, c]))
                    except ValueError:
                        pass
            auc = float(np.mean(auc_scores)) if auc_scores else 0.0

        rows.append({
            "collection": str(coll),
            "recordings": int(sub_n),
            "macro_f1": f1,
            "top1_acc": top1,
            "top3_acc": top3,
            "roc_auc": auc,
        })

    return pd.DataFrame(rows)

