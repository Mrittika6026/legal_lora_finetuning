#!/bin/bash
# Quick start script for LoRA fine-tuning on 3090

set -e

echo "=========================================="
echo "LoRA Fine-tuning Setup for 3090 (24GB)"
echo "=========================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "[1/5] Setting up Python path..."
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
echo "✓ PYTHONPATH set to: $PYTHONPATH"
echo ""

echo "[2/5] Checking dependencies..."
if ! python -c "import torch; import transformers; import peft; import wandb" 2>/dev/null; then
    echo "⚠ Some dependencies missing. Installing..."
    pip install -r requirements.txt
else
    echo "✓ All dependencies installed"
fi
echo ""

echo "[3/5] Checking W&B login..."
if wandb status &>/dev/null; then
    echo "✓ W&B is logged in"
else
    echo "⚠ W&B not logged in. Please run: wandb login"
    echo "   Get your API key from: https://wandb.ai/settings"
fi
echo ""

echo "[4/5] Checking data files..."
if [ -f "data/train_instruct.jsonl" ] && [ -f "data/val_instruct.jsonl" ]; then
    TRAIN_LINES=$(wc -l < data/train_instruct.jsonl)
    VAL_LINES=$(wc -l < data/val_instruct.jsonl)
    echo "✓ Training data: $TRAIN_LINES lines"
    echo "✓ Validation data: $VAL_LINES lines"
else
    echo "⚠ Data files not found in data/ directory"
fi
echo ""

echo "[5/5] Ready to train!"
echo ""
echo "To start training, run:"
echo "  python -m lora_finetune.main train --config configs/qwen3-4b.yaml"
echo ""
echo "Or:"
echo "  export PYTHONPATH=$PROJECT_ROOT/src:\$PYTHONPATH"
echo "  python scripts/train.py train --config configs/qwen3-4b.yaml"
echo ""
echo "View training progress at: https://wandb.ai"
echo "=========================================="

