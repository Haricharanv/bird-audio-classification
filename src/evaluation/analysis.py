"""Evaluation analysis: confusion matrix, error auditing, domain shift demo, and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_per_collection_metrics,
)
from src.utils.io import save_json
from src.utils.logging import get_logger

logger = get_logger("bird_classification.analysis")


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    max_classes_to_display: int = 50,
) -> None:
    """Plot and save normalized confusion matrix as a PNG heatmap.
    
    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.
        class_names: List of class names.
        save_path: Output PNG path.
        max_classes_to_display: If class count exceeds this, displays top frequent classes.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Normalize rows (true labels)
    with np.errstate(all="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # Subsample if too many classes for clear visual display
    num_classes = len(class_names)
    if num_classes > max_classes_to_display:
        counts = np.bincount(y_true, minlength=num_classes)
        top_indices = np.argsort(counts)[::-1][:max_classes_to_display]
        cm_norm = cm_norm[np.ix_(top_indices, top_indices)]
        display_names = [class_names[i] for i in top_indices]
    else:
        display_names = class_names

    plt.figure(figsize=(max(8, len(display_names) * 0.35), max(6, len(display_names) * 0.35)))
    sns.heatmap(
        cm_norm,
        annot=len(display_names) <= 20,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True,
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Species")
    plt.ylabel("True Species")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}.")


def find_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_k: int = 10,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Identify top most confused species pairs (True != Predicted).
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        top_k: Number of confused pairs to return.
        save_path: Optional destination CSV.
        
    Returns:
        DataFrame cataloging confused species pairs and counts.
    """
    pairs: Dict[tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pair = (class_names[t], class_names[p])
            pairs[pair] = pairs.get(pair, 0) + 1

    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    records = [
        {"true_species": pair[0], "predicted_species": pair[1], "error_count": count}
        for pair, count in sorted_pairs
    ]
    df = pd.DataFrame(records)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved top {len(df)} confused species pairs to {save_path}.")
    return df


def save_confident_misclassifications(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str],
    frontend: Any,
    out_dir: Union[str, Path],
    top_n: int = 20,
) -> pd.DataFrame:
    """Save spectrogram plots for the 20 most-confident model misclassifications.
    
    Args:
        df_meta: Metadata DataFrame.
        y_true: True class indices.
        y_pred_probs: Predicted probability matrix.
        class_names: Species class names.
        frontend: Feature extractor to compute mel spectrogram for visualization.
        out_dir: Destination folder.
        top_n: Maximum number of error plots to produce.
        
    Returns:
        DataFrame of audited error clips.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_classes = np.argmax(y_pred_probs, axis=1)
    max_probs = np.max(y_pred_probs, axis=1)

    errors = []
    for idx in range(len(y_true)):
        t = y_true[idx]
        p = pred_classes[idx]
        if t != p:
            errors.append({
                "meta_idx": idx,
                "filename": df_meta.iloc[idx]["filename"],
                "true_species": class_names[t],
                "pred_species": class_names[p],
                "confidence": float(max_probs[idx]),
                "collection": df_meta.iloc[idx].get("collection", "XC"),
            })

    if not errors:
        logger.info("Zero misclassifications found; skipping error spectrogram plotting.")
        return pd.DataFrame()

    errors_df = pd.DataFrame(errors).sort_values("confidence", ascending=False).head(top_n)
    errors_df.to_csv(out_dir / "confident_errors.csv", index=False)

    import torch
    from src.data.preprocess import load_audio

    for rank, (_, row) in enumerate(errors_df.iterrows(), start=1):
        fn = str(row["filename"])
        true_sp = row["true_species"]
        pred_sp = row["pred_species"]
        conf = row["confidence"]

        # Attempt to load audio
        audio_dir = Path(frontend.cfg.data.raw_dir) / "train_audio" if hasattr(frontend, "cfg") else Path("data/raw/train_audio")
        audio_path = audio_dir / fn
        if not audio_path.is_file():
            audio_path = audio_dir / true_sp / fn

        if not audio_path.is_file():
            continue

        try:
            waveform = load_audio(audio_path, target_sr=32000, trim_silence=False)
            waveform_t = torch.from_numpy(waveform[:160000]).unsqueeze(0)
            spec = frontend(waveform_t).squeeze().cpu().numpy()

            plt.figure(figsize=(9, 4))
            plt.imshow(spec, aspect="auto", origin="lower", cmap="magma")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Rank #{rank}: True: {true_sp} | Pred: {pred_sp} ({conf:.1%} conf)\nFile: {fn}")
            plt.xlabel("Time Frames")
            plt.ylabel("Mel Bins")
            plt.tight_layout()
            plt.savefig(out_dir / f"error_{rank:02d}_{Path(fn).stem}.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"Could not render error plot for {fn}: {e}")

    logger.info(f"Generated {len(errors_df)} confident misclassification plots in {out_dir}.")
    return errors_df


def generate_full_evaluation_report(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    df_meta: pd.DataFrame,
    class_to_idx: Dict[str, int],
    output_dir: Union[str, Path],
    frontend: Optional[Any] = None,
    is_multilabel: bool = False,
) -> Dict[str, Any]:
    """Orchestrate end-to-end evaluation, metrics, collection breakdowns, and diagnostic plots.
    
    Args:
        y_true: Ground truth targets.
        y_pred_probs: Predicted probabilities (N, C).
        df_meta: Evaluation split metadata DataFrame.
        class_to_idx: Species to index mapping.
        output_dir: Directory where report artifacts are saved.
        frontend: Optional frontend module for spectrogram visualization.
        is_multilabel: Multilabel flag.
        
    Returns:
        Dictionary of final scalar metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_to_class = {idx: sp for sp, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # 1. Compute overall metrics
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred_probs=y_pred_probs,
        is_multilabel=is_multilabel,
        class_names=class_names,
    )
    per_class_df = metrics.pop("per_class_report")
    per_class_df.to_csv(out_dir / "per_class_report.csv", index=False)

    # 2. Compute collection source breakdown
    if "collection" in df_meta.columns:
        coll_df = compute_per_collection_metrics(
            df_meta=df_meta,
            y_true=y_true,
            y_pred_probs=y_pred_probs,
            is_multilabel=is_multilabel,
        )
        coll_df.to_csv(out_dir / "collection_breakdown.csv", index=False)
        metrics["per_collection"] = coll_df.to_dict(orient="records")

    # Save metrics JSON
    save_json(metrics, out_dir / "metrics.json")
    logger.info(f"Saved summary metrics to {out_dir / 'metrics.json'}.")

    if not is_multilabel:
        pred_labels = np.argmax(y_pred_probs, axis=1)

        # 3. Normalized Confusion Matrix
        plot_and_save_confusion_matrix(
            y_true=y_true,
            y_pred=pred_labels,
            class_names=class_names,
            save_path=out_dir / "confusion_matrix.png",
        )

        # 4. Top-10 Confused Pairs
        find_top_confused_pairs(
            y_true=y_true,
            y_pred=pred_labels,
            class_names=class_names,
            top_k=10,
            save_path=out_dir / "top10_confused_pairs.csv",
        )

        # 5. Top-20 Confident Misclassifications
        if frontend is not None:
            save_confident_misclassifications(
                df_meta=df_meta,
                y_true=y_true,
                y_pred_probs=y_pred_probs,
                class_names=class_names,
                frontend=frontend,
                out_dir=out_dir / "top20_misclassified",
            )

    return metrics

