from typing import Dict, Any

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from datasets import load_dataset, Dataset, DatasetDict

from .config import DataConfig
from .tokenization import apply_chat_template, tokenize_with_mask


def _load_pretty_jsonl(path: str, max_records: Optional[int] = None) -> Dataset:
    """Parse pretty-printed JSON objects separated by blank lines, with optional early stop."""
    data: List[Dict[str, Any]] = []
    buffer: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and len(data) >= max_records:
                break
            if not line.strip() and not buffer:
                continue
            buffer.append(line)
            try:
                obj = json.loads("".join(buffer))
            except json.JSONDecodeError:
                continue
            else:
                data.append(obj)
                buffer = []
    if buffer:
        try:
            data.append(json.loads("".join(buffer)))
        except json.JSONDecodeError:
            pass
    return Dataset.from_list(data)


def _safe_load_dataset(data_cfg: DataConfig, sample_size: Optional[int] = None) -> DatasetDict:
    # Robust fallback: parse pretty-printed JSON objects without relying on pyarrow inference.
    train_ds = _load_pretty_jsonl(data_cfg.train_file, max_records=sample_size)
    if data_cfg.eval_file:
        val_ds = _load_pretty_jsonl(data_cfg.eval_file, max_records=sample_size)
        return DatasetDict({"train": train_ds, "validation": val_ds})
    return DatasetDict({"train": train_ds})


def load_and_prepare_dataset(
    data_cfg: DataConfig,
    tokenizer,
    assistant_token: str,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    dataset = _safe_load_dataset(data_cfg, sample_size=sample_size)

    def _format_chat(example):
        text = apply_chat_template(tokenizer, example["messages"])
        return {"text": text}

    dataset = dataset.map(
        _format_chat,
        remove_columns=dataset["train"].column_names
    )

    def _tokenize(example):
        return tokenize_with_mask(
            tokenizer=tokenizer,
            text=example["text"],
            max_length=data_cfg.max_length,
            assistant_token=assistant_token,
        )

    dataset = dataset.map(
        _tokenize,
        batched=False,
        remove_columns=["text"]
    )
    return dataset

