.PHONY: help install prepare prepare-debug train-debug train-baseline train-effnet train-crnn train-ast evaluate test seeds ablations app clean

help:
	@echo "=========================================================================="
	@echo "BirdCLEF+ 2025 Bioacoustic Audio Classification Workflow"
	@echo "=========================================================================="
	@echo "install         : Install project dependencies from requirements.txt"
	@echo "prepare         : Preprocess metadata, splits, noise bank, and chunk cache"
	@echo "prepare-debug   : Preprocess 5 debug species for rapid development"
	@echo "train-debug     : Run 1-epoch baseline training on CPU (debug mode)"
	@echo "train-baseline  : Train Baseline CNN architecture"
	@echo "train-effnet    : Train EfficientNet-B0 pretrained model"
	@echo "train-crnn      : Train CRNN (CNN + BiGRU + Attention Pooling)"
	@echo "train-ast       : Train Audio Spectrogram Transformer (AST)"
	@echo "evaluate        : Run recording-level evaluation on held-out test split"
	@echo "test            : Execute unit test suite with pytest"
	@echo "seeds           : Run multi-seed reproducibility benchmark (3 seeds)"
	@echo "ablations       : Run systematic ablation studies from configs/ablations.yaml"
	@echo "app             : Launch Gradio web application on localhost:7860"
	@echo "clean           : Remove cache files, logs, and temp artifacts"
	@echo "=========================================================================="

install:
	pip install -r requirements.txt

prepare:
	python scripts/prepare_data.py --config configs/base.yaml

prepare-debug:
	python scripts/prepare_data.py --config configs/base.yaml debug=true

train-debug:
	python scripts/train.py --config configs/baseline_cnn.yaml debug=true

train-baseline:
	python scripts/train.py --config configs/baseline_cnn.yaml

train-effnet:
	python scripts/train.py --config configs/effnet_b0.yaml

train-crnn:
	python scripts/train.py --config configs/crnn.yaml

train-ast:
	python scripts/train.py --config configs/ast.yaml

evaluate:
	@checkpoint=$$(find runs -name "best_model.pt" | sort -r | head -n 1); \
	if [ -z "$$checkpoint" ]; then \
		echo "No checkpoint found! Train a model first with 'make train-debug'"; exit 1; \
	fi; \
	echo "Evaluating checkpoint: $$checkpoint"; \
	python scripts/evaluate.py --checkpoint $$checkpoint --config configs/base.yaml --split test

test:
	pytest -v

seeds:
	python scripts/run_seeds.py --config configs/baseline_cnn.yaml debug=true

ablations:
	python scripts/run_ablations.py --config configs/ablations.yaml debug=true

app:
	python app/app.py

clean:
	rm -rf .pytest_cache `find . -name "__pycache__"` `find . -name "*.pyc"`

