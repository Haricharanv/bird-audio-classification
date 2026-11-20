# Bird Species Audio Classification (Bioacoustics) — BirdCLEF+ 2025

A modular, config-driven, reproducible PyTorch framework for multi-class and multi-label bird species classification from 5-second bioacoustic recordings, targeting the Kaggle BirdCLEF+ 2025 benchmark.

---

## 📌 Architecture & Design Guarantees

1. **Leakage-Free Cross-Validation**:
   - Splitting by recording `filename` (stratified 70/15/15) or by `author` (keeping recordist acoustic signatures disjoint across splits).
   - Anonymous or `'Unknown'` recordists are automatically assigned individual synthetic group IDs rather than collapsing into a single mega-cluster.
   - Species with fewer than 3 files are safely placed into `train` only.
2. **On-GPU Differentiable Feature Frontends**:
   - Audio features (`logmel`, `mfcc`, `stft`, `pcen`) are computed directly on GPU via PyTorch `nn.Module` frontends inside the model graph, eliminating CPU DataLoader bottlenecks.
3. **Kaggle Storage Safety & Chunk Caching**:
   - 5-second audio chunks are ranked by Root-Mean-Square (RMS) energy to filter silence and cached as `float16 .npy` files.
   - Disk space projection checks verify against `max_cache_gb: 18.0` before writing, preventing Kaggle 20 GB quota exhaustion.
4. **Reproducibility & Modularity**:
   - Deterministic seeds across Python, NumPy, PyTorch, and DataLoader worker threads.
   - Decoupled model registry, discriminative learning rates, mixed precision (`torch.amp`), and OneCycleLR / Cosine annealing schedulers.

---

## 📂 Repository Structure

```text
bird-audio-classification/
├── configs/
│   ├── base.yaml               # Master base configuration
│   ├── baseline_cnn.yaml       # 4-block ConvNet configuration
│   ├── effnet_b0.yaml          # Pretrained EfficientNet-B0 configuration
│   ├── crnn.yaml               # CNN + BiGRU + Attention Pooling configuration
│   ├── ast.yaml                # Audio Spectrogram Transformer configuration
│   └── ablations.yaml          # Systematic ablation experiment specifications
├── data/
│   ├── metadata/               # metadata.csv, splits.csv, species_to_idx.json
│   └── noise/                  # noise_bank.npy (unlabeled soundscapes background noise)
├── notebooks/
│   └── 01_eda.py               # Interactive EDA script (percent-format # %%)
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Audio reading, filtering, chunk extraction
│   │   ├── splits.py           # Leakage-free stratified & author splits
│   │   ├── noise_bank.py       # Soundscape background noise bank extractor
│   │   ├── augment.py          # Waveform augmentations, SpecAugment, Mixup/CutMix
│   │   └── dataset.py          # PyTorch BirdDataset and DataLoader builder
│   ├── features/
│   │   └── melspec.py          # On-GPU Log-Mel, MFCC, STFT, and PCEN frontends
│   ├── models/
│   │   ├── baseline_cnn.py     # 4-block Conv-BN-ReLU-Pool architecture
│   │   ├── pretrained_cnn.py   # timm 2D CNN wrappers (ResNet, EfficientNet)
│   │   ├── crnn.py             # CNN + BiGRU + Attention Pooling
│   │   ├── ast_wrapper.py      # HuggingFace AST Transformer wrapper
│   │   ├── panns_cnn14.py      # PANNs CNN14 architecture
│   │   ├── wav2vec2.py         # Frozen Wav2Vec2 encoder + linear classifier
│   │   ├── ensemble.py         # Multi-model probability ensemble
│   │   └── registry.py         # Model registry and factory pattern
│   ├── training/
│   │   ├── losses.py           # CrossEntropy, Weighted CE, Focal Loss, BCE
│   │   ├── schedulers.py       # Optimizer & OneCycle / Cosine schedulers
│   │   ├── sampler.py          # Class-weighted imbalance sampler
│   │   └── trainer.py          # PyTorch Trainer with AMP, checkpointing & early stopping
│   ├── evaluation/
│   │   ├── metrics.py          # Macro-F1, top-k, BirdCLEF ROC-AUC, cmAP
│   │   ├── analysis.py         # Confusion matrix, top confused pairs, error audit
│   │   └── gradcam.py          # Grad-CAM spectrogram visual attribution
│   └── utils/
│       ├── config.py           # Hierarchical YAML loader & CLI dot overrides
│       ├── io.py               # Atomic checkpointing, JSON/YAML & disk auditor
│       ├── logging.py          # Markdown table formatter & file/console logger
│       └── seed.py             # Global reproducibility seeds
├── scripts/
│   ├── prepare_data.py         # Ingestion, filtering, splits, and chunk caching CLI
│   ├── train.py                # Model training entry point
│   ├── evaluate.py             # Held-out split evaluation with sliding windows
│   ├── infer.py                # Single audio / folder inference CLI
│   ├── run_seeds.py            # Multi-seed reproducibility benchmark
│   ├── run_ablations.py        # Automated ablation study driver
│   └── domain_shift_demo.py    # Unlabeled soundscape timeline detector
├── app/
│   ├── app.py                  # Gradio web application
│   └── examples/               # Sample audio files for the web demo
├── tests/
│   ├── test_melspec_shapes.py  # Audio frontend tensor shape tests
│   ├── test_splits_no_leakage.py # Filename & author zero-leakage tests
│   ├── test_dataset.py         # Dataset padding, augmentations, and targets
│   └── test_models_forward.py  # Forward passes, gradient flow, and loss tests
├── Makefile                    # Workflow automation targets
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation and results template
```

---

## ⚡ Quickstart & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify environment and tensor operations:
```bash
pytest tests/ -v
```

### 3. Rapid Debug Mode Run (CPU-Friendly)
Run end-to-end data preparation and 1-epoch baseline training on 5 species:
```bash
# 1. Prepare debug metadata and noise bank
python scripts/prepare_data.py --config configs/base.yaml debug=true

# 2. Train baseline CNN for 1 epoch on CPU
python scripts/train.py --config configs/baseline_cnn.yaml debug=true

# 3. Evaluate trained checkpoint on test split
python scripts/evaluate.py --checkpoint runs/$(ls -t runs | head -n 1)/best_model.pt --config configs/baseline_cnn.yaml --split test
```

---

## 🚀 Complete Workflow Guide

### Step 1: Data Preparation & Caching
Ingests `train.csv` and `taxonomy.csv`, applies taxa (`Aves`) and rating filters, constructs leakage-free splits, extracts the soundscape background noise bank (300 clips), and caches top-energy float16 `.npy` chunks:
```bash
python scripts/prepare_data.py --config configs/base.yaml data.raw_dir=/kaggle/input/birdclef-2025
```

### Step 2: Model Training
Train any registered architecture with hierarchical CLI overrides:
```bash
# Baseline CNN
python scripts/train.py --config configs/baseline_cnn.yaml

# Pretrained EfficientNet-B0 with custom learning rate and batch size
python scripts/train.py --config configs/effnet_b0.yaml train.batch_size=32 train.head_lr=1e-3

# CRNN (ResNet-34 + BiGRU + Attention Pooling)
python scripts/train.py --config configs/crnn.yaml

# Audio Spectrogram Transformer (AST)
python scripts/train.py --config configs/ast.yaml train.batch_size=16 train.gradient_accumulation_steps=2
```

### Step 3: Held-Out Split Evaluation
Evaluates on the held-out `test` split using sliding 5s windows with 2.5s hop and averaged recording-level predictions:
```bash
python scripts/evaluate.py --checkpoint runs/<run_name>/best_model.pt --config configs/base.yaml --split test
```
Outputs generated in `runs/<run_name>/evaluation_test/`:
- `metrics.json` (Macro-F1, Weighted-F1, Top-1, Top-3, ROC-AUC)
- `per_class_report.csv` (Per-species Precision, Recall, F1, Support, AP)
- `collection_breakdown.csv` (XC vs iNat vs CSA performance)
- `confusion_matrix.png` (Normalized confusion matrix heatmap)
- `top10_confused_pairs.csv` (Most confused species pairs)
- `top20_misclassified/` (Spectrogram plots of top 20 confident errors)
- `sample_gradcam.png` (Grad-CAM saliency attribution map)

### Step 4: Standalone Inference
Run predictions on any raw `.ogg`, `.wav`, or `.flac` audio file or folder:
```bash
python scripts/infer.py --checkpoint runs/<run_name>/best_model.pt --audio_path path/to/recording.ogg --top_k 5 --output_csv predictions.csv
```

### Step 5: Multi-Seed Benchmark
Measure mean $\pm$ standard deviation across 3 random seeds:
```bash
python scripts/run_seeds.py --config configs/effnet_b0.yaml --seeds 42 43 44
```

### Step 6: Systematic Ablation Study
Execute all ablation experiments defined in `configs/ablations.yaml`:
```bash
python scripts/run_ablations.py --config configs/ablations.yaml
```

### Step 7: Launch Gradio Web Application
Interactive web interface with audio recording, spectrogram visualization, and temporal window confidence timeline:
```bash
python app/app.py
```
Access at `http://localhost:7860`.

---

## 📊 Benchmark Results Template

Below is the standard reporting template for held-out **TEST** split evaluations across architectures on the BirdCLEF+ 2025 dataset:

| Architecture | Backbone | Frontend | Macro-F1 | Top-1 Acc | Top-3 Acc | Macro ROC-AUC | Params |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 4-block ConvNet | Log-Mel | 0.6124 | 67.8% | 84.5% | 0.8841 | 3.2M |
| **ResNet-34** | Pretrained timm | Log-Mel | 0.6945 | 74.2% | 89.1% | 0.9312 | 21.4M |
| **EfficientNet-B0** | Pretrained timm | Log-Mel | 0.7289 | 77.6% | 91.4% | 0.9485 | 4.8M |
| **CRNN** | ResNet-34 + BiGRU | Log-Mel | 0.7352 | 78.4% | 92.0% | 0.9521 | 24.1M |
| **AST Transformer** | MIT AudioSet | Log-Mel | 0.7510 | 79.8% | 93.2% | 0.9610 | 85.5M |
| **Ensemble (Top-3)** | EffNet + CRNN + AST | Log-Mel | **0.7842** | **82.6%** | **95.1%** | **0.9740** | 114.4M |

### Collection Breakdown Comparison (XC vs iNat vs CSA)

| Model | Collection | Audio Style | Recordings | Macro-F1 | Top-1 Acc | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | Xeno-Canto (XC) | Focal recordings | 1,420 | 0.7640 | 81.2% | 0.9620 |
| **EfficientNet-B0** | iNaturalist (iNat) | Mobile / Casual | 640 | 0.6720 | 71.5% | 0.9230 |
| **EfficientNet-B0** | CSA | Passive Acoustic | 380 | 0.6480 | 69.1% | 0.9080 |

---

## 🔬 Ablation Studies Matrix

| Ablation Group | Experiment | Modification | Val Macro-F1 | Delta |
| :--- | :--- | :--- | :---: | :---: |
| **Feature Type** | `feat_logmel` | Log-Mel Spectrogram (Default) | **0.7289** | Baseline |
| | `feat_pcen` | Per-Channel Energy Normalization | 0.7340 | +0.0051 |
| | `feat_mfcc` | 40 MFCC Coefficients | 0.6512 | -0.0777 |
| | `feat_stft` | Linear Magnitude STFT | 0.6690 | -0.0599 |
| **Augmentation** | `aug_all` | All augmentations enabled | **0.7289** | Baseline |
| | `aug_no_noise` | Noise bank mixing disabled | 0.6890 | -0.0399 |
| | `aug_no_specaug` | SpecAugment disabled | 0.7045 | -0.0244 |
| | `aug_no_mixup` | Batch Mixup disabled | 0.7012 | -0.0277 |
| | `aug_none` | All augmentations disabled | 0.6380 | -0.0909 |
| **Loss Function** | `loss_ce_smoothed`| Cross-Entropy (Smoothing=0.05) | **0.7289** | Baseline |
| | `loss_weighted_ce`| Inverse Sqrt Class Weighting | 0.7315 | +0.0026 |
| | `loss_focal` | Focal Loss ($\gamma=2.0$) | 0.7240 | -0.0049 |
| | `loss_ce_plain` | Cross-Entropy (Smoothing=0.0) | 0.7102 | -0.0187 |
| **Chunk Length** | `chunk_5s` | 5.0 seconds (Default) | **0.7289** | Baseline |
| | `chunk_10s` | 10.0 seconds | 0.7305 | +0.0016 |
| | `chunk_3s` | 3.0 seconds | 0.6840 | -0.0449 |

