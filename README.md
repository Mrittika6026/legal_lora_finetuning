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
```
pip install -U transformers accelerate datasets peft bitsandbytes sentencepiece
export PYTHONPATH=$(pwd)/lora_finetune/src:$PYTHONPATH

# Optional smoke test (fast, no training)
python -m lora_finetune.scripts.smoke_test --config lora_finetune/configs/qwen3-4b.yaml
# Add --check-model to also load the base model and do a tiny forward (GPU needed)

# Train
python -m lora_finetune.main train --config lora_finetune/configs/qwen3-4b.yaml

# Infer (after training)
python -m lora_finetune.main infer --config lora_finetune/configs/qwen3-4b.yaml --prompt "Explain how tax exemption approval continues."
```

To change tokenization, edit `tokenizer` or `assistant_token` in the YAML. To train/replace a tokenizer, point `tokenizer.name` to your artifact (and optionally update `max_length`). To adjust LoRA scope, tweak `lora.target_modules`.

