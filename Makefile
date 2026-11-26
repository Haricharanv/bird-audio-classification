.PHONY: help install download-indian16 download-india50 prepare prepare-indian prepare-debug train-debug train-baseline train-effnet train-indian train-crnn train-ast evaluate test seeds ablations app clean

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

help:
	@echo "=========================================================================="
	@echo "Bird Species Bioacoustic Classification Workflow"
	@echo "=========================================================================="
	@echo "install           : Install project dependencies from requirements.txt"
	@echo "download-indian16 : Download 16 iconic Indian bird species (20 clips each)"
	@echo "download-india50  : Auto-fetch top 50 Indian birds from iNaturalist API"
	@echo "prepare-indian    : Prepare splits, noise bank, and train cache (indian16)"
	@echo "train-indian      : Train EfficientNet-B0 on indian16 config"
	@echo "prepare           : Preprocess metadata, splits, noise bank, and chunk cache"
	@echo "prepare-debug     : Preprocess 5 debug species for rapid development"
	@echo "train-debug       : Run 1-epoch baseline training on CPU (debug mode)"
	@echo "train-baseline    : Train Baseline CNN architecture"
	@echo "train-effnet      : Train EfficientNet-B0 pretrained model"
	@echo "train-crnn        : Train CRNN (CNN + BiGRU + Attention Pooling)"
	@echo "train-ast         : Train Audio Spectrogram Transformer (AST)"
	@echo "evaluate          : Run recording-level evaluation on newest checkpoint"
	@echo "test              : Execute unit test suite with pytest"
	@echo "seeds             : Run multi-seed reproducibility benchmark (3 seeds)"
	@echo "ablations         : Run systematic ablation studies from configs/ablations.yaml"
	@echo "app               : Launch 2-Tab Gradio web app (BirdNET + Custom Models)"
	@echo "clean             : Remove cache files, logs, and temp artifacts"
	@echo "=========================================================================="

install:
	$(PYTHON) -m pip install -r requirements.txt

download-indian16:
	$(PYTHON) scripts/download_inat_birds.py --out_dir ../data/indian_birds --preset indian16 --per_species 20 --workers 8

download-india50:
	$(PYTHON) scripts/download_inat_birds.py --out_dir ../data/india_top_50 --auto_india_top 50 --per_species 30 --workers 8

prepare-indian:
	$(PYTHON) scripts/prepare_data.py --config configs/indian_birds.yaml debug=false

train-indian:
	$(PYTHON) scripts/train.py --config configs/indian_birds.yaml

prepare:
	$(PYTHON) scripts/prepare_data.py --config configs/base.yaml

prepare-debug:
	$(PYTHON) scripts/prepare_data.py --config configs/base.yaml debug=true

train-debug:
	$(PYTHON) scripts/train.py --config configs/baseline_cnn.yaml debug=true

train-baseline:
	$(PYTHON) scripts/train.py --config configs/baseline_cnn.yaml

train-effnet:
	$(PYTHON) scripts/train.py --config configs/effnet_b0.yaml

train-crnn:
	$(PYTHON) scripts/train.py --config configs/crnn.yaml

train-ast:
	$(PYTHON) scripts/train.py --config configs/ast.yaml

evaluate:
	@checkpoint=$$(ls -t runs/*/best_model.pt 2>/dev/null | head -n 1); \
	if [ -z "$$checkpoint" ]; then \
		echo "No checkpoint found! Train a model first with 'make train-indian' or 'make train-debug'"; exit 1; \
	fi; \
	echo "Evaluating checkpoint: $$checkpoint"; \
	$(PYTHON) scripts/evaluate.py --checkpoint "$$checkpoint" --config configs/base.yaml --split test

test:
	$(PYTHON) -m pytest tests/ -v

seeds:
	$(PYTHON) scripts/run_seeds.py --config configs/baseline_cnn.yaml debug=true

ablations:
	$(PYTHON) scripts/run_ablations.py --config configs/ablations.yaml debug=true

app:
	$(PYTHON) app/app.py

clean:
	rm -rf .pytest_cache `find . -name "__pycache__"` `find . -name "*.pyc"`
