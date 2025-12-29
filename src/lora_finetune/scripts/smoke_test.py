"""
Lightweight smoke test to validate config, tokenizer, and dataset mapping.
Run from repo root (ensure PYTHONPATH points to lora_finetune/src):

python -m lora_finetune.scripts.smoke_test --config lora_finetune/configs/qwen3-4b.yaml
"""

import argparse
import sys
from typing import Optional

from lora_finetune.config import load_config
from lora_finetune.data import load_and_prepare_dataset
from lora_finetune.model import load_base_model
from lora_finetune.tokenization import load_tokenizer


def smoke_test(config_path: str, check_model: bool = False, sample_size: int | None = None):
    cfg = load_config(config_path)
    print(f"[ok] loaded config: {config_path}")

    tokenizer = load_tokenizer(cfg.tokenizer)
    print(f"[ok] loaded tokenizer: {cfg.tokenizer.name}")

    dataset = load_and_prepare_dataset(
        data_cfg=cfg.data,
        tokenizer=tokenizer,
        assistant_token=cfg.tokenizer.assistant_token,
        sample_size=sample_size,
    )
    sample = dataset["train"][0]
    label_count = sum(1 for x in sample["labels"] if x != -100)
    print(f"[ok] dataset mapping: input_ids={len(sample['input_ids'])}, labels_active={label_count}")

    if check_model:
        model = load_base_model(cfg.model, cfg.quantization)
        # Trigger a small forward pass shape check with dummy ids from sample
        import torch

        input_ids = torch.tensor([sample["input_ids"][:8]], device=model.device)
        attn = torch.ones_like(input_ids)
        _ = model(input_ids=input_ids, attention_mask=attn)
        print(f"[ok] model load + dummy forward on device {model.device}")


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="LoRA smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default="lora_finetune/configs/qwen3-4b.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Also load base model and run a tiny forward pass (requires GPU for 4-bit).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of examples per split for a quick smoke test.",
    )
    args = parser.parse_args(argv)

    try:
        smoke_test(args.config, check_model=args.check_model, sample_size=args.sample_size)
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

