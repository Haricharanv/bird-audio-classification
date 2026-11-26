"""Gradio Web Application for Bird Species Audio Classification with Grad-CAM and Model Switching."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_audio
from src.evaluation.gradcam import GradCAM
from src.features.melspec import build_frontend
from src.models.ensemble import EnsembleModel, load_ensemble_from_checkpoints
from src.models.registry import build_model
from src.utils.audio_inference import aggregate_window_probs, slice_sliding_windows
from src.utils.config import Config, load_config
from src.utils.io import load_checkpoint
from src.utils.logging import get_logger

logger = get_logger("gradio_app")

# Global model cache keyed by checkpoint selection
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
ENSEMBLE_LABEL = "Ensemble (All Compatible Models)"
BIRDNET_LABEL = "BirdNET v2.4 (6,500+ Global & Indian Species)"


def _is_birdnet_available() -> bool:
    """Return True when the birdnet package is installed in the active environment."""
    return importlib.util.find_spec("birdnet") is not None


def _format_birdnet_species(raw_name: str) -> Tuple[str, str, str]:
    """Parse BirdNET 'Scientific Name_Common Name' into (display_name, common_name, sci_name)."""
    raw = str(raw_name).strip()
    if "_" in raw:
        sci, common = raw.split("_", 1)
        sci, common = sci.strip(), common.strip()
        display = f"{common} ({sci})" if common and sci else (common or sci)
        return display, (common or sci), sci
    return raw, raw, ""


def _resolve_taxonomy_map(
    cfg: Config,
    ckpt_dir: Optional[Path] = None,
) -> Dict[str, Tuple[str, str, str]]:
    """Locate taxonomy.csv and return mapping: primary_label -> (display_name, common_name, sci_name)."""
    candidate_paths: List[Path] = []
    if ckpt_dir:
        candidate_paths.append(ckpt_dir / "taxonomy.csv")
    raw_dir = getattr(getattr(cfg, "data", None), "raw_dir", None)
    if raw_dir:
        raw_path = Path(raw_dir)
        if not raw_path.is_absolute():
            raw_path = (PROJECT_ROOT / raw_path).resolve()
        candidate_paths.append(raw_path / "taxonomy.csv")
    candidate_paths.extend([
        PROJECT_ROOT / "taxonomy.csv",
        PROJECT_ROOT.parent / "data" / "indian_birds" / "taxonomy.csv",
        PROJECT_ROOT.parent / "data" / "real_birds" / "taxonomy.csv",
        PROJECT_ROOT / "data" / "metadata_indian" / "taxonomy.csv",
        PROJECT_ROOT / "data" / "metadata_real" / "taxonomy.csv",
        PROJECT_ROOT / "data" / "metadata" / "taxonomy.csv",
    ])

    tax_map: Dict[str, Tuple[str, str, str]] = {}
    for tax_path in candidate_paths:
        if tax_path.is_file():
            try:
                tax_df = pd.read_csv(tax_path, dtype=str)
                if "primary_label" in tax_df.columns and "common_name" in tax_df.columns:
                    for _, row in tax_df.iterrows():
                        lbl = str(row["primary_label"]).strip()
                        common = str(row.get("common_name", "")).strip()
                        sci = str(row.get("scientific_name", "")).strip()
                        if common and common.lower() != "nan":
                            sci_clean = sci if sci and sci.lower() != "nan" else ""
                            display = f"{common} ({sci_clean})" if sci_clean else common
                            tax_map[lbl] = (display, common, sci_clean)
                    if tax_map:
                        break
            except Exception as e:
                logger.warning(f"Could not read taxonomy file {tax_path}: {e}")
    return tax_map


def load_species_display_names(
    cfg: Config,
    class_names: list[str],
    ckpt_dir: Optional[Path] = None,
) -> Tuple[list[str], list[str], list[list[str]]]:
    """Map class labels to human-readable Common Name (Scientific Name) and table rows."""
    tax_map = _resolve_taxonomy_map(cfg, ckpt_dir=ckpt_dir)
    full_names = list(class_names)
    short_names = list(class_names)
    table_rows: List[List[str]] = []

    for i, lbl in enumerate(class_names):
        if lbl in tax_map:
            full_names[i], short_names[i], sci = tax_map[lbl]
            table_rows.append([short_names[i], sci or "-", lbl])
        else:
            table_rows.append([lbl, "-", lbl])

    return full_names, short_names, table_rows


def discover_checkpoints() -> List[Path]:
    """Return all best_model.pt checkpoints sorted newest-first."""
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(list(runs_dir.glob("**/best_model.pt")), key=os.path.getmtime, reverse=True)


def find_default_checkpoint() -> Optional[Path]:
    """Search for the most recent checkpoint in runs/."""
    ckpts = discover_checkpoints()
    return ckpts[0] if ckpts else None


def get_model_choices() -> List[str]:
    """Build dropdown options for available run checkpoints, ensemble, and BirdNET v2.4."""
    ckpts = discover_checkpoints()
    choices = [p.parent.name for p in ckpts]
    if len(choices) >= 2:
        choices.insert(1, ENSEMBLE_LABEL)
    if _is_birdnet_available():
        insert_idx = min(2, len(choices)) if ENSEMBLE_LABEL in choices else min(1, len(choices))
        choices.insert(insert_idx, BIRDNET_LABEL)
    return choices or ["Latest"]


def _load_single_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load one checkpoint into standardized metadata dict."""
    logger.info(f"Loading checkpoint for Gradio app: {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(ckpt.get("config", {}))
    if not cfg:
        cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    display_names, short_names, table_rows = load_species_display_names(
        cfg, class_names, ckpt_dir=ckpt_path.parent
    )

    num_classes = len(class_names)
    cfg.model.num_classes = num_classes
    if "model" in cfg:
        cfg.model.pretrained = False

    model = build_model(cfg, num_classes=num_classes)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.to(device)
    model.eval()

    return {
        "kind": "pytorch",
        "model": model,
        "class_names": class_names,
        "display_names": display_names,
        "short_names": short_names,
        "species_rows": table_rows,
        "device": device,
        "cfg": cfg,
        "run_dir": ckpt_path.parent,
    }


def _load_ensemble_entry(device: torch.device) -> Dict[str, Any]:
    """Load all checkpoints sharing the newest model's exact species vocabulary into an EnsembleModel."""
    ckpts = discover_checkpoints()
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in runs/ for ensemble.")

    ref_ckpt = torch.load(ckpts[0], map_location="cpu")
    ref_classes = ref_ckpt.get("class_to_idx", {})
    compatible_paths: List[Path] = []
    for p in ckpts:
        c = torch.load(p, map_location="cpu")
        if c.get("class_to_idx", {}) == ref_classes:
            compatible_paths.append(p)

    if len(compatible_paths) == 1:
        return _load_single_checkpoint(compatible_paths[0], device)

    ref_cfg = Config(ref_ckpt.get("config", {})) or load_config(PROJECT_ROOT / "configs" / "base.yaml")
    ensemble = load_ensemble_from_checkpoints(compatible_paths, ref_cfg, device=device)
    idx_to_class = {v: k for k, v in ref_classes.items()}
    class_names = [idx_to_class[i] for i in range(len(ref_classes))]
    display_names, short_names, table_rows = load_species_display_names(
        ref_cfg, class_names, ckpt_dir=compatible_paths[0].parent
    )

    return {
        "kind": "pytorch",
        "model": ensemble,
        "class_names": class_names,
        "display_names": display_names,
        "short_names": short_names,
        "species_rows": table_rows,
        "device": device,
        "cfg": ref_cfg,
        "run_dir": compatible_paths[0].parent,
        "member_names": [p.parent.name for p in compatible_paths],
    }


def _load_birdnet_entry() -> Dict[str, Any]:
    """Load pretrained Cornell BirdNET v2.4 LiteRT backend directly in-process (6,522 species)."""
    import birdnet

    logger.info("Loading BirdNET Acoustic v2.4 LiteRT model (6,522 species)...")
    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    backend = model.backend_type(
        model_path=model.model_path,
        device_name="CPU",
        half_precision=False,
        **model.backend_kwargs,
    )
    backend.load()

    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    frontend = build_frontend(cfg)
    frontend.eval()
    species_list = list(model.species_list)
    return {
        "kind": "birdnet",
        "model": model,
        "backend": backend,
        "lock": threading.Lock(),
        "sample_rate": int(model.get_sample_rate()),
        "segment_sec": float(model.get_segment_size_s()),
        "frontend": frontend,
        "class_names": species_list,
        "display_names": ["6,522 Global & Indian Bird Species (BirdNET v2.4)"],
        "short_names": ["BirdNET v2.4"],
        "species_rows": [
            ["6,522 Global & Indian Bird Species", "Cornell BirdNET Acoustic v2.4 (LiteRT)", "global_6522"]
        ],
        "device": torch.device("cpu"),
        "cfg": cfg,
        "run_dir": None,
    }


def get_model_entry(selection: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache selected model, ensemble, or BirdNET entry."""
    ckpt_env = os.environ.get("CHECKPOINT_PATH")
    if selection == BIRDNET_LABEL:
        cache_key = BIRDNET_LABEL
        target_path: Optional[Path] = None
    elif selection == ENSEMBLE_LABEL:
        cache_key = ENSEMBLE_LABEL
        target_path = None
    elif selection is None or selection == "Latest":
        if ckpt_env and Path(ckpt_env).is_file():
            cache_key = str(Path(ckpt_env).resolve())
            target_path = Path(ckpt_env)
        else:
            target_path = find_default_checkpoint()
            if target_path is None:
                if _is_birdnet_available():
                    cache_key = BIRDNET_LABEL
                else:
                    raise FileNotFoundError(
                        "No trained model checkpoint found! Train a model first via `python scripts/train.py` "
                        "or set the CHECKPOINT_PATH environment variable."
                    )
            else:
                cache_key = target_path.parent.name
    else:
        cache_key = selection
        target_path = PROJECT_ROOT / "runs" / selection / "best_model.pt"

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cache_key == BIRDNET_LABEL:
        entry = _load_birdnet_entry()
    elif cache_key == ENSEMBLE_LABEL:
        entry = _load_ensemble_entry(device)
    else:
        if target_path is None or not target_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {target_path}")
        entry = _load_single_checkpoint(target_path, device)

    MODEL_CACHE[cache_key] = entry
    return entry


def get_model_and_classes(selection: Optional[str] = None) -> Tuple[Any, list[str], torch.device]:
    """Backwards-compatible accessor returning (model, display_names, device)."""
    entry = get_model_entry(selection)
    return entry["model"], entry["display_names"], entry["device"]


def get_species_table_for_selection(selection: Optional[str] = None) -> List[List[str]]:
    """Return supported species rows [Common Name, Scientific Name, Label] for the UI table."""
    try:
        entry = get_model_entry(selection)
        return entry["species_rows"]
    except Exception:
        return []


def _find_gradcam_model(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Return a single Conv2d-equipped model instance suitable for Grad-CAM."""
    raw = model.module if hasattr(model, "module") else model
    if isinstance(raw, EnsembleModel):
        for m in raw.models:
            if any(isinstance(sub, torch.nn.Conv2d) for sub in m.modules()):
                return m
        return None
    if any(isinstance(sub, torch.nn.Conv2d) for sub in raw.modules()):
        return raw
    return None


def _build_notice_figure(message: str, title: str) -> plt.Figure:
    """Create a clean centered informational matplotlib figure."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, wrap=True)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_spectrogram(spec: np.ndarray, title: str) -> plt.Figure:
    """Plot a Log-Mel spectrogram figure."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def _plot_timeline(
    timestamps: List[float] | np.ndarray,
    top_conf: List[float] | np.ndarray,
    top_labels: List[str],
    confidence_threshold: float,
) -> plt.Figure:
    """Plot sliding-window top predicted species confidence over time."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(timestamps, top_conf, marker="o", color="#1f77b4", linewidth=2)
    ax.fill_between(timestamps, top_conf, alpha=0.15, color="#1f77b4")
    ax.axhline(confidence_threshold, color="#d9534f", linestyle=":", linewidth=1.2, label="Threshold")

    for t, conf, label in zip(timestamps, top_conf, top_labels):
        ax.annotate(
            f"{label}\n({conf:.0%})",
            (t, conf),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("Per-Window Top Bird Species Timeline")
    ax.set_xlabel("Recording Timestamp (seconds)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0.0, 1.18)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def _build_gradcam_figure(
    model: torch.nn.Module,
    waveform_window: torch.Tensor,
    pred_idx: int,
    pred_label: str,
) -> plt.Figure:
    """Compute Grad-CAM saliency overlay on the active 5s window, or return a fallback plot."""
    cam_model = _find_gradcam_model(model)
    if cam_model is None:
        return _build_notice_figure(
            "Grad-CAM unavailable for this architecture",
            "Grad-CAM Visual Attribution",
        )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    try:
        gradcam = GradCAM(cam_model)
        cam, spec, _ = gradcam.generate_cam(waveform_window.unsqueeze(0), target_class_idx=pred_idx)
        gradcam.remove_hooks()

        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", alpha=0.65)
        im = ax.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.45)
        ax.set_title(f"Grad-CAM Focus: {pred_label}")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")
        fig.colorbar(im, ax=ax, label="Relevance")
        fig.tight_layout()
    except Exception as e:
        logger.warning(f"Grad-CAM generation skipped: {e}")
        plt.close(fig)
        return _build_notice_figure(f"Grad-CAM skipped: {e}", "Grad-CAM Visual Attribution")
    return fig


def _predict_birdnet(
    entry: Dict[str, Any],
    audio_filepath: str,
    confidence_threshold: float,
) -> Tuple[str, Dict[str, float], plt.Figure, plt.Figure, plt.Figure]:
    """Run in-process BirdNET v2.4 LiteRT inference across 3s windows with spectrogram and timeline."""
    cfg = entry["cfg"]
    bn_sr = int(entry["sample_rate"])
    seg_sec = float(entry["segment_sec"])
    spec_sr = int(getattr(getattr(cfg, "audio", None), "sample_rate", 32000))
    win_samples = int(spec_sr * float(getattr(getattr(cfg, "audio", None), "duration_sec", 5.0)))

    wf_bn = load_audio(audio_filepath, target_sr=bn_sr, trim_silence=False, peak_norm=False)
    windows_bn, timestamps = slice_sliding_windows(
        wf_bn, sample_rate=bn_sr, window_sec=seg_sec, hop_sec=seg_sec
    )
    with entry["lock"]:
        logits = entry["backend"].predict(np.ascontiguousarray(windows_bn, dtype=np.float32))
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    mean_probs = aggregate_window_probs(probs, windows=windows_bn, mode="confidence_energy")
    species_list = entry["class_names"]
    top_indices = np.argsort(mean_probs)[::-1][:3]
    top_3_dict = {
        _format_birdnet_species(species_list[idx])[0]: float(mean_probs[idx])
        for idx in top_indices
    }

    top_idx = int(top_indices[0])
    top_prob = float(mean_probs[top_idx])
    top_display = _format_birdnet_species(species_list[top_idx])[0]
    best_win_idx = int(np.argmax(np.max(probs, axis=1)))
    best_start_sec = float(timestamps[best_win_idx])

    fig_gradcam = _build_notice_figure(
        "Grad-CAM attribution is available for our custom PyTorch CNN / CRNN / AST models.\n"
        "Switch 'Model / Ensemble' to a trained checkpoint above to view Grad-CAM.",
        "Grad-CAM Attribution (Custom PyTorch Models)",
    )

    if top_prob < confidence_threshold:
        status_md = (
            f"> ⚠️ **Low Confidence ({top_prob:.1%} < {confidence_threshold:.0%} threshold)** — "
            f"BirdNET v2.4's closest match across **{len(species_list):,} global species** is **{top_display}**."
        )
    else:
        status_md = (
            f"> ✅ **Detected: {top_display}** — Confidence **{top_prob:.1%}** "
            f"across **{len(windows_bn)}** window(s) via **BirdNET v2.4** ({len(species_list):,} species)."
        )

    wf_spec = load_audio(audio_filepath, target_sr=spec_sr, trim_silence=False)
    center_sample = int((best_start_sec + seg_sec / 2.0) * spec_sr)
    start_sample = max(0, min(max(0, len(wf_spec) - win_samples), center_sample - win_samples // 2))
    slice_wf = wf_spec[start_sample: start_sample + win_samples]
    if len(slice_wf) < win_samples:
        slice_wf = np.pad(slice_wf, (0, win_samples - len(slice_wf)), mode="constant")
    with torch.no_grad():
        spec = entry["frontend"](torch.from_numpy(slice_wf).float().unsqueeze(0)).squeeze().cpu().numpy()
    fig_spec = _plot_spectrogram(spec, f"5-Second Log-Mel Spectrogram (Peak @ {best_start_sec:.1f}s)")

    top_per_win_idx = np.argmax(probs, axis=1)
    top_per_win_conf = np.max(probs, axis=1)
    top_per_win_labels = [_format_birdnet_species(species_list[i])[1] for i in top_per_win_idx]
    fig_timeline = _plot_timeline(timestamps, top_per_win_conf, top_per_win_labels, confidence_threshold)

    return status_md, top_3_dict, fig_spec, fig_gradcam, fig_timeline


def predict_bird_audio(
    audio_filepath: str,
    model_selection: Optional[str] = None,
    confidence_threshold: float = 0.35,
) -> Tuple[str, Dict[str, float], plt.Figure, plt.Figure, plt.Figure]:
    """Run model inference on uploaded/recorded audio, computing spectrogram, Grad-CAM, and timeline."""
    empty_fig = plt.figure()
    if not audio_filepath:
        return "Upload or record a bird audio clip to begin.", {}, empty_fig, plt.figure(), plt.figure()

    entry = get_model_entry(model_selection)
    if entry.get("kind") == "birdnet":
        return _predict_birdnet(entry, audio_filepath, confidence_threshold)

    model = entry["model"]
    class_names = entry["display_names"]
    short_names = entry["short_names"]
    device = entry["device"]
    cfg = entry["cfg"]

    sr = int(getattr(getattr(cfg, "audio", None), "sample_rate", 32000))
    window_sec = float(getattr(getattr(cfg, "audio", None), "duration_sec", 5.0))
    hop_sec = float(getattr(getattr(cfg, "audio", None), "val_hop_sec", 2.5))

    waveform = load_audio(audio_filepath, target_sr=sr, trim_silence=False)
    windows_np, timestamps = slice_sliding_windows(
        waveform, sample_rate=sr, window_sec=window_sec, hop_sec=hop_sec
    )
    windows_tensor = torch.from_numpy(windows_np).float().to(device)

    with torch.no_grad():
        out = model(windows_tensor)
        if isinstance(model, EnsembleModel):
            probs = out.cpu().numpy()
        else:
            probs = torch.softmax(out, dim=-1).cpu().numpy()

    mean_probs = aggregate_window_probs(probs, windows=windows_np, mode="confidence_energy")
    top_indices = np.argsort(mean_probs)[::-1][:3]
    top_3_dict = {class_names[idx]: float(mean_probs[idx]) for idx in top_indices}

    top_idx = int(top_indices[0])
    top_prob = float(mean_probs[top_idx])
    ensemble_note = (
        f" using {len(entry['member_names'])}-model ensemble"
        if "member_names" in entry
        else ""
    )

    if top_prob < confidence_threshold:
        status_md = (
            f"> ⚠️ **Low Confidence ({top_prob:.1%} < {confidence_threshold:.0%} threshold)** — "
            f"This clip may be ambient noise or a bird species outside the **{len(class_names)} trained classes**. "
            f"Closest match: **{class_names[top_idx]}**."
        )
    else:
        status_md = (
            f"> ✅ **Detected: {class_names[top_idx]}** — Confidence **{top_prob:.1%}** "
            f"across **{len(windows_np)}** window(s){ensemble_note}."
        )

    best_win_idx = int(np.argmax(np.max(probs, axis=1)))
    active_window = windows_tensor[best_win_idx]

    raw_model = model.module if hasattr(model, "module") else model
    fe_model = raw_model.models[0] if isinstance(raw_model, EnsembleModel) else raw_model
    with torch.no_grad():
        spec = fe_model.frontend(active_window.unsqueeze(0)).squeeze().cpu().numpy()

    fig_spec = _plot_spectrogram(spec, f"5-Second Log-Mel Spectrogram (Window @ {timestamps[best_win_idx]:.1f}s)")
    fig_gradcam = _build_gradcam_figure(
        model=model,
        waveform_window=active_window,
        pred_idx=top_idx,
        pred_label=short_names[top_idx],
    )

    top_per_window_idx = np.argmax(probs, axis=1)
    top_per_window_conf = np.max(probs, axis=1)
    top_per_window_labels = [short_names[idx] for idx in top_per_window_idx]
    fig_timeline = _plot_timeline(timestamps, top_per_window_conf, top_per_window_labels, confidence_threshold)

    return status_md, top_3_dict, fig_spec, fig_gradcam, fig_timeline


def _empty_per_class_df() -> pd.DataFrame:
    """Return an empty DataFrame matching per_class_report.csv headers."""
    return pd.DataFrame(columns=["species", "precision", "recall", "f1", "support", "average_precision"])


def _build_all_runs_comparison_markdown() -> str:
    """Read evaluation_test/metrics.json across all runs to render a dynamic comparison table."""
    ckpts = discover_checkpoints()
    rows: List[str] = []
    for ckpt in ckpts:
        run_dir = ckpt.parent
        metrics_file = run_dir / "evaluation_test" / "metrics.json"
        if metrics_file.is_file():
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                rows.append(
                    f"| `{run_dir.name}` | {m.get('macro_f1', 0.0):.4f} | "
                    f"{m.get('weighted_f1', 0.0):.4f} | {m.get('top1_accuracy', 0.0):.2%} | "
                    f"{m.get('top3_accuracy', 0.0):.2%} | {m.get('roc_auc', 0.0):.4f} |"
                )
            except Exception as e:
                logger.warning(f"Could not parse metrics at {metrics_file}: {e}")

    if not rows:
        return (
            "### Evaluated Runs in `runs/`\n"
            "_No `evaluation_test/metrics.json` reports found yet. "
            "Run `python scripts/evaluate.py --checkpoint <path> --split test` to generate test benchmarks._"
        )

    header = (
        "### All Evaluated Models (Held-Out Test Split)\n\n"
        "| Run Directory | Macro-F1 | Weighted-F1 | Top-1 Acc | Top-3 Acc | ROC-AUC |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )
    return header + "\n".join(rows)


def _resolve_benchmark_run_dir(selection: Optional[str]) -> Optional[Path]:
    """Map UI model selection to the corresponding run directory on disk."""
    if selection == BIRDNET_LABEL:
        return None
    if selection in (None, "Latest", ENSEMBLE_LABEL):
        for ckpt in discover_checkpoints():
            if (ckpt.parent / "evaluation_test" / "metrics.json").is_file():
                return ckpt.parent
        default_ckpt = find_default_checkpoint()
        return default_ckpt.parent if default_ckpt else None
    candidate = PROJECT_ROOT / "runs" / selection
    return candidate if candidate.is_dir() else None


def get_benchmarks_for_selection(
    selection: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str], pd.DataFrame]:
    """Load held-out test metrics, confusion matrix, Grad-CAM, and per-class report for the selected run."""
    comparison_md = _build_all_runs_comparison_markdown()

    if selection == BIRDNET_LABEL:
        md = (
            "### BirdNET v2.4 Pretrained Reference Model\n"
            "- **Coverage:** 6,522 global bird species (including Indian avifauna) via LiteRT acoustic backbone.\n"
            "- **Evaluation Note:** BirdNET v2.4 is an external pretrained model; per-dataset held-out confusion matrices "
            "apply to your custom-trained PyTorch models below.\n\n"
            + comparison_md
        )
        return md, None, None, _empty_per_class_df()

    run_dir = _resolve_benchmark_run_dir(selection)
    if run_dir is None:
        return "No evaluated model run found in `runs/`.\n\n" + comparison_md, None, None, _empty_per_class_df()

    eval_dir = run_dir / "evaluation_test"
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        md = (
            f"### Selected Run: `{run_dir.name}`\n"
            f"_No `evaluation_test/metrics.json` found for `{run_dir.name}`. "
            f"Run `python scripts/evaluate.py --checkpoint runs/{run_dir.name}/best_model.pt --split test`._\n\n"
            + comparison_md
        )
        return md, None, None, _empty_per_class_df()

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected_md = (
        f"### Selected Run Benchmarks: `{run_dir.name}`\n"
        f"- **Macro-F1:** `{m.get('macro_f1', 0.0):.4f}` | "
        f"**Weighted-F1:** `{m.get('weighted_f1', 0.0):.4f}` | "
        f"**Top-1 Accuracy:** `{m.get('top1_accuracy', 0.0):.2%}` | "
        f"**Top-3 Accuracy:** `{m.get('top3_accuracy', 0.0):.2%}` | "
        f"**ROC-AUC:** `{m.get('roc_auc', 0.0):.4f}`\n\n"
        + comparison_md
    )

    cm_path = eval_dir / "confusion_matrix.png"
    gc_path = eval_dir / "sample_gradcam.png"
    per_class_path = eval_dir / "per_class_report.csv"

    per_class_df = _empty_per_class_df()
    if per_class_path.is_file():
        per_class_df = pd.read_csv(per_class_path).round(4)

    return (
        selected_md,
        str(cm_path) if cm_path.is_file() else None,
        str(gc_path) if gc_path.is_file() else None,
        per_class_df,
    )


def create_demo_audio_examples(examples_dir: Path) -> list[list[str]]:
    """Return real held-out test clips if available, else synthetic demo clips."""
    candidate_dirs = [
        PROJECT_ROOT.parent / "data" / "indian_birds" / "demo_examples",
        PROJECT_ROOT / "demo_examples",
        PROJECT_ROOT.parent / "data" / "real_birds" / "demo_examples",
    ]
    for real_demo_dir in candidate_dirs:
        if real_demo_dir.is_dir():
            real_files = sorted(real_demo_dir.glob("*.ogg"))
            if real_files:
                return [[str(p)] for p in real_files]

    examples_dir.mkdir(parents=True, exist_ok=True)
    ex1 = examples_dir / "sample_bird_chirp_1.ogg"
    ex2 = examples_dir / "sample_bird_chirp_2.ogg"

    if not ex1.is_file():
        sr = 32000
        t = np.linspace(0, 5, sr * 5)
        f0, f1 = 2000, 6000
        chirp = 0.5 * np.sin(2 * np.pi * (f0 * t + ((f1 - f0) / 10) * (t**2 % 1.0)))
        sf.write(str(ex1), chirp.astype(np.float32), sr)

    if not ex2.is_file():
        sr = 32000
        t = np.linspace(0, 8, sr * 8)
        chirp2 = 0.4 * np.sin(2 * np.pi * 3500 * t) + 0.1 * np.random.randn(len(t))
        sf.write(str(ex2), chirp2.astype(np.float32), sr)

    return [[str(ex1)], [str(ex2)]]


def _on_model_change(selection: Optional[str]) -> Tuple[List[List[str]], str, Optional[str], Optional[str], pd.DataFrame]:
    """Refresh supported species table and benchmark artifacts when model selection changes."""
    rows = get_species_table_for_selection(selection)
    bm_md, cm_path, gc_path, pc_df = get_benchmarks_for_selection(selection)
    return rows, bm_md, cm_path, gc_path, pc_df


def build_app():
    import gradio as gr

    examples_dir = PROJECT_ROOT / "app" / "examples"
    examples = create_demo_audio_examples(examples_dir)
    model_choices = get_model_choices()
    default_choice = model_choices[0]
    initial_species = get_species_table_for_selection(default_choice)
    init_bm_md, init_cm, init_gc, init_pc = get_benchmarks_for_selection(default_choice)

    with gr.Blocks(title="Bird Species Audio Classifier | Bioacoustics AI") as demo:
        gr.Markdown(
            "# 🐦 Bird Species Audio Classification (Bioacoustics AI)\n"
            "Identify bird calls with our **custom-trained PyTorch models (EfficientNet-B0 / CRNN / AST / Ensemble)** "
            "or switch to **BirdNET v2.4 (6,500+ Global & Indian Species)**, inspect **Log-Mel Spectrograms** & "
            "**Grad-CAM**, and review **Held-Out Test Benchmarks**."
        )

        with gr.Tabs():
            with gr.Tab("🎙️ Live Bird Classifier"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Input Audio Recording (.ogg, .wav, .flac, .mp3)",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=default_choice,
                            label="Model / Ensemble",
                        )
                        conf_slider = gr.Slider(
                            minimum=0.10,
                            maximum=0.80,
                            value=0.35,
                            step=0.05,
                            label="Unknown-Species Confidence Threshold",
                        )
                        predict_btn = gr.Button("Classify Audio", variant="primary")
                        gr.Examples(examples=examples, inputs=audio_input)

                        with gr.Accordion("Supported Bird Species Coverage", open=False):
                            species_table = gr.Dataframe(
                                value=initial_species,
                                headers=["Common Name", "Scientific Name", "Class ID"],
                                interactive=False,
                            )

                    with gr.Column(scale=1):
                        status_output = gr.Markdown("Ready. Upload or select an audio example and click **Classify Audio**.")
                        label_output = gr.Label(num_top_classes=3, label="Top-3 Predicted Species")
                        gradcam_output = gr.Plot(label="Grad-CAM Visual Attribution (Where the AI Listened)")
                        spec_output = gr.Plot(label="Log-Mel Spectrogram")
                        timeline_output = gr.Plot(label="Sliding Window Confidence Timeline")

            with gr.Tab("📊 Evaluation & Benchmarks"):
                benchmark_md = gr.Markdown(value=init_bm_md)
                with gr.Row():
                    cm_image = gr.Image(value=init_cm, type="filepath", label="Held-Out Test Confusion Matrix")
                    eval_gradcam_image = gr.Image(value=init_gc, type="filepath", label="Sample Test Grad-CAM Explanation")
                per_class_df = gr.Dataframe(
                    value=init_pc,
                    label="Per-Species Test Report (Precision / Recall / F1 / AP)",
                    interactive=False,
                )

        model_dropdown.change(
            fn=_on_model_change,
            inputs=model_dropdown,
            outputs=[species_table, benchmark_md, cm_image, eval_gradcam_image, per_class_df],
        )

        predict_btn.click(
            fn=predict_bird_audio,
            inputs=[audio_input, model_dropdown, conf_slider],
            outputs=[status_output, label_output, spec_output, gradcam_output, timeline_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    allowed = [
        str(p)
        for p in (
            PROJECT_ROOT / "runs",
            PROJECT_ROOT.parent / "data" / "indian_birds",
            PROJECT_ROOT.parent / "data" / "real_birds",
        )
        if p.is_dir()
    ] or None
    share_flag = os.environ.get("GRADIO_SHARE", "0").strip().lower() in ("1", "true", "yes")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share_flag, allowed_paths=allowed)
