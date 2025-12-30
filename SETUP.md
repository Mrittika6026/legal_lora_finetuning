# Quick Start Guide - LoRA Fine-tuning on 3090 (24GB VRAM)

## Prerequisites
- NVIDIA RTX 3090 with 24GB VRAM
- Python 3.8+
- CUDA installed

## Step 1: Install Dependencies

```bash
cd /workspace/legal_lora_finetuning
pip install -r requirements.txt
```

## Step 2: Setup Weights & Biases (W&B)

1. **Login to W&B** (you'll need your API key):
   ```bash
   wandb login
   ```
   Enter your API key when prompted. You can get it from https://wandb.ai/settings

2. **Verify W&B is working**:
   ```bash
   wandb status
   ```

## Step 3: Set Python Path

```bash
export PYTHONPATH=/workspace/legal_lora_finetuning/src:$PYTHONPATH
```

Or add it to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
echo 'export PYTHONPATH=/workspace/legal_lora_finetuning/src:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Verify Data Files

Check that your data files exist:
```bash
ls -lh data/train_instruct.jsonl data/val_instruct.jsonl
```

## Step 5: (Optional) Test Configuration

Run a quick smoke test to verify everything is set up correctly:
```bash
python -m lora_finetune.scripts.smoke_test --config configs/qwen3-4b.yaml
```

## Step 6: Start Training

```bash
python -m lora_finetune.main train --config configs/qwen3-4b.yaml
```

Or using the script wrapper:
```bash
python scripts/train.py
```

## Monitoring Training

- **W&B Dashboard**: Training metrics will be logged automatically. View at https://wandb.ai
- **Local Outputs**: Checkpoints and adapters will be saved to `outputs/qwen3-4b-lora/`

## Configuration Notes for 24GB VRAM

The current config (`configs/qwen3-4b.yaml`) is optimized for your 3090:
- **4-bit quantization**: Disabled (24GB is sufficient for Qwen3-4B)
- **Batch size**: 1 per device with 16 gradient accumulation steps (effective batch size = 16)
- **Max sequence length**: 2048 tokens
- **bf16**: Enabled for efficient training

If you encounter OOM (Out of Memory) errors, you can:
1. Enable 4-bit quantization: Set `quantization.load_in_4bit: true` in the config
2. Reduce `max_length` to 1024 or 1536
3. Reduce `gradient_accumulation_steps` to 8

## Resume Training

If training is interrupted, you can resume from a checkpoint:
```bash
python -m lora_finetune.main train --config configs/qwen3-4b.yaml --resume-from-checkpoint outputs/qwen3-4b-lora/checkpoint-500
```

## Run Inference After Training

```bash
python -m lora_finetune.main infer --config configs/qwen3-4b.yaml --prompt "Your legal question here"
```

## Troubleshooting

### CUDA Out of Memory
- Enable 4-bit quantization in config
- Reduce `max_length` in data config
- Reduce `per_device_train_batch_size` (already at 1, but you can try)

### W&B Not Logging
- Verify `wandb login` was successful
- Check that `report_to: "wandb"` in training config
- Ensure wandb section exists in YAML config

### Import Errors
- Make sure PYTHONPATH is set correctly
- Verify all packages in requirements.txt are installed

