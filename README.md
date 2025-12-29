LoRA fine-tune project scaffold for Qwen/Qwen3-4B (Hugging Face, PEFT, Accelerate, bitsandbytes). The code is modular so you can swap tokenization, data, or hyperparameters via YAML without touching the training loop.

### Data Size 

Train sample for LoRA: 10k QA pair - almost

weight and Biases ; hyperparameter tuning - MLFLow

Training: 5589
Val: 294





### Layout
- `configs/` – YAML configs (model, tokenizer, data, LoRA, training, generation).
- `data/` – training/validation JSONL stored locally (`train_instruct.jsonl`, `val_instruct.jsonl`).
- `src/lora_finetune/` – reusable modules (config, tokenization, data, model, training, inference, CLI).
- `scripts/` – thin wrappers to run training or inference.
- `notebooks/` – lightweight demo/notes notebook pointing to the scripts.
- `outputs/` (runtime) – adapters, logs, checkpoints (ignored via `.gitignore` entry below).

### Quickstart

#### Setup
```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

#### Weights & Biases (W&B) Setup
W&B is integrated for experiment tracking and hyperparameter tuning:

1. **Install and login**:
   ```bash
   pip install wandb
   wandb login  # Get your API key from https://wandb.ai/settings
   ```

2. **Configure in YAML**: Edit `configs/qwen3-4b.yaml` to set your W&B project:
   ```yaml
   wandb:
     project: "legal-lora-qwen3"  # Your W&B project name
     entity: null  # Optional: your W&B username/team
     name: "qwen3-4b-lora-r32"  # Optional: custom run name
     tags: ["legal", "qwen3", "lora"]
   ```

3. **View runs**: Training metrics, hyperparameters, and system stats are automatically logged. View them at https://wandb.ai

#### Training
```bash
# Optional smoke test (fast, no training)
python -m lora_finetune.scripts.smoke_test --config configs/qwen3-4b.yaml
# Add --check-model to also load the base model and do a tiny forward (GPU needed)

# Train (with W&B tracking)
python -m lora_finetune.main train --config configs/qwen3-4b.yaml

# Infer (after training)
python -m lora_finetune.main infer --config configs/qwen3-4b.yaml --prompt "Explain how tax exemption approval continues."
```

To change tokenization, edit `tokenizer` or `assistant_token` in the YAML. To train/replace a tokenizer, point `tokenizer.name` to your artifact (and optionally update `max_length`). To adjust LoRA scope, tweak `lora.target_modules`.

