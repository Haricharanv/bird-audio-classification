"""Regression tests for sliding-window pooling, caching, sampler alignment, ensemble loading, and Gradio app."""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.data.dataset import create_dataloaders
from src.data.preprocess import cache_energy_chunks
from src.models.baseline_cnn import BaselineCNN
from src.models.ensemble import EnsembleModel, load_ensemble_from_checkpoints
from src.utils.audio_inference import aggregate_window_probs, slice_sliding_windows
from src.utils.config import Config
from src.utils.io import save_checkpoint


def test_slice_and_confidence_energy_pooling():
    sr = 32000
    # 10s waveform: first 5s quiet ambient, last 5s loud chirp
    quiet = np.random.randn(sr * 5).astype(np.float32) * 0.01
    loud = np.sin(2 * np.pi * 3000 * np.linspace(0, 5, sr * 5)).astype(np.float32) * 0.9
    waveform = np.concatenate([quiet, loud])

    windows, timestamps = slice_sliding_windows(waveform, sample_rate=sr, window_sec=5.0, hop_sec=2.5)
    assert windows.shape == (3, 160000)
    assert timestamps == [0.0, 2.5, 5.0]

    # Quiet window guesses class 0 with low confidence; loud window predicts class 1 with 0.95 confidence
    probs = np.array([
        [0.55, 0.45],
        [0.50, 0.50],
        [0.05, 0.95],
    ], dtype=np.float32)
    agg = aggregate_window_probs(probs, windows=windows, mode="confidence_energy")
    assert agg[1] > agg[0], "Active loud bird call window should outweigh quiet ambient windows"


def test_cache_energy_chunks_train_only_and_weighted_sampler(tmp_path):
    raw_dir = tmp_path / "raw"
    audio_dir = raw_dir / "train_audio"
    audio_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)

    sr = 32000
    tone = (0.5 * np.sin(2 * np.pi * 2000 * np.linspace(0, 6, sr * 6))).astype(np.float32)
    rows = []
    for sp, idx, split_name in [("sp_a", 0, "train"), ("sp_a", 1, "train"), ("sp_a", 2, "val"), ("sp_b", 0, "train"), ("sp_b", 1, "test")]:
        fn = f"{sp}_{idx}.ogg"
        sf.write(str(audio_dir / fn), tone, sr)
        rows.append({
            "primary_label": sp,
            "secondary_labels": [],
            "filename": fn,
            "author": f"auth_{sp}_{idx}",
            "collection": "iNat",
            "target": 0 if sp == "sp_a" else 1,
            "split": split_name,
        })

    splits_df = pd.DataFrame(rows)
    cfg = Config({
        "task": "multiclass",
        "system": {"num_workers": 0, "pin_memory": False},
        "data": {
            "raw_dir": str(raw_dir),
            "cache_dir": str(cache_dir),
            "metadata_dir": str(meta_dir),
            "max_chunks_per_species": 50,
            "max_cache_gb": 1.0,
        },
        "audio": {"sample_rate": 32000, "duration_sec": 5.0, "val_hop_sec": 2.5, "trim_silence": False, "peak_normalize": True},
        "augment": {"waveform": {"noise_bank_prob": 0.0, "gain_prob": 0.0, "pitch_shift_prob": 0.0, "time_stretch_prob": 0.0, "time_shift_prob": 0.0}},
        "train": {"batch_size": 2, "use_weighted_sampler": True},
    })

    chunks_df = cache_energy_chunks(splits_df, cfg)
    cached_files = set(chunks_df["filename"].tolist())
    assert "sp_a_2.ogg" not in cached_files, "Val recordings should not be cached"
    assert "sp_b_1.ogg" not in cached_files, "Test recordings should not be cached"
    assert {"sp_a_0.ogg", "sp_a_1.ogg", "sp_b_0.ogg"}.issubset(cached_files)

    train_loader, _, _ = create_dataloaders(
        cfg=cfg,
        splits_df=splits_df,
        species_to_idx={"sp_a": 0, "sp_b": 1},
        cached_chunks_df=chunks_df,
    )
    assert len(train_loader.sampler) == len(train_loader.dataset)
    batch = next(iter(train_loader))
    assert batch["waveform"].shape == (2, 160000)


def test_load_ensemble_from_dict_config_checkpoints(tmp_path):
    cfg = Config({
        "task": "multiclass",
        "audio": {"feature_type": "logmel", "sample_rate": 32000, "n_fft": 1024, "hop_length": 320, "win_length": 1024, "n_mels": 128},
        "augment": {"spec": {"spec_aug_prob": 0.0}},
        "model": {"name": "baseline_cnn", "in_chans": 1, "num_classes": 3, "dropout": 0.1, "pretrained": False},
    })
    class_to_idx = {"bird_a": 0, "bird_b": 1, "bird_c": 2}
    ckpts = []
    for i in range(2):
        model = BaselineCNN(cfg, num_classes=3)
        p = tmp_path / f"model_{i}.pt"
        save_checkpoint(p, model=model, epoch=1, metric_val=0.5, class_to_idx=class_to_idx, cfg=cfg)
        ckpts.append(p)

    ens = load_ensemble_from_checkpoints(ckpts, cfg, device="cpu")
    assert isinstance(ens, EnsembleModel)
    out = ens(torch.randn(1, 160000))
    assert out.shape == (1, 3)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-4)


def test_fetch_india_top_species_parsing(monkeypatch):
    import io
    import json
    from scripts.download_inat_birds import fetch_india_top_species

    sample_payload = {
        "results": [
            {"taxon": {"name": "Eudynamys scolopaceus", "preferred_common_name": "Asian Koel"}},
            {"taxon": {"name": "Orthotomus sutorius", "preferred_common_name": "Common Tailorbird"}},
            {"taxon": {"name": "Pavo cristatus", "preferred_common_name": ""}},
        ]
    }

    class _FakeResp:
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResp(json.dumps(sample_payload).encode("utf-8")),
    )
    species = fetch_india_top_species(top_n=2)
    assert len(species) == 2
    assert species[0] == ("asian_koel", "Asian Koel", "Eudynamys scolopaceus")
    assert species[1] == ("common_tailorbird", "Common Tailorbird", "Orthotomus sutorius")


def test_app_birdnet_formatting_and_benchmarks(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import app.app as bird_app

    assert bird_app._format_birdnet_species("Eudynamys scolopaceus_Asian Koel") == (
        "Asian Koel (Eudynamys scolopaceus)",
        "Asian Koel",
        "Eudynamys scolopaceus",
    )

    fake_runs = tmp_path / "runs"
    run_dir = fake_runs / "20260101_effnet_b0_test"
    eval_dir = run_dir / "evaluation_test"
    eval_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"stub")
    (eval_dir / "metrics.json").write_text(json.dumps({
        "macro_f1": 0.75,
        "weighted_f1": 0.78,
        "top1_accuracy": 0.80,
        "top3_accuracy": 0.95,
        "roc_auc": 0.91,
    }))
    pd.DataFrame([
        {"species": "asian_koel", "precision": 0.9, "recall": 0.8, "f1": 0.847, "support": 5, "average_precision": 0.88}
    ]).to_csv(eval_dir / "per_class_report.csv", index=False)
    (eval_dir / "confusion_matrix.png").write_bytes(b"png")

    monkeypatch.setattr(bird_app, "PROJECT_ROOT", tmp_path)
    md, cm_path, gc_path, per_class_df = bird_app.get_benchmarks_for_selection("20260101_effnet_b0_test")
    assert "0.7500" in md
    assert "80.00%" in md
    assert cm_path == str(eval_dir / "confusion_matrix.png")
    assert gc_path is None
    assert len(per_class_df) == 1
    assert per_class_df.iloc[0]["species"] == "asian_koel"


def test_load_audio_resamples_non_32k_without_resampy(tmp_path):
    import numpy as np
    import soundfile as sf
    from src.data.preprocess import load_audio

    wav_path = tmp_path / "phone_44k.wav"
    t = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
    sf.write(str(wav_path), 0.5 * np.sin(2 * np.pi * 2500.0 * t), 44100)

    wf_32k = load_audio(wav_path, target_sr=32000, trim_silence=False)
    wf_48k = load_audio(wav_path, target_sr=48000, trim_silence=False, peak_norm=False)
    assert wf_32k.ndim == 1 and abs(len(wf_32k) - 32000) <= 2
    assert wf_48k.ndim == 1 and abs(len(wf_48k) - 48000) <= 2
    assert wf_32k.dtype == np.float32
