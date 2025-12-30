from typing import Dict, Any

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict

from .config import DataConfig
from .tokenization import apply_chat_template, tokenize_with_mask


# def _load_pretty_jsonl(path: str, max_records: Optional[int] = None) -> Dataset:
#     """Parse JSONL file (one JSON object per line) or pretty-printed JSON objects separated by blank lines."""
#     data: List[Dict[str, Any]] = []
#     buffer: List[str] = []
#     file_path = Path(path)
    
#     # Count total lines for progress bar
#     total_lines = sum(1 for _ in file_path.open("r", encoding="utf-8"))
    
#     with file_path.open("r", encoding="utf-8") as f:
#         for line in tqdm(f, total=total_lines, desc=f"Loading {file_path.name}", unit="lines"):
#             if max_records is not None and len(data) >= max_records:
#                 break
            
#             stripped_line = line.strip()
            
#             # Skip empty lines if buffer is empty (standard JSONL case)
#             if not stripped_line:
#                 if buffer:
#                     # Empty line with buffer means end of multi-line JSON
#                     try:
#                         obj = json.loads("".join(buffer))
#                         data.append(obj)
#                         buffer = []
#                     except json.JSONDecodeError:
#                         buffer = []  # Clear invalid buffer
#                 continue
            
#             # Try parsing the line directly first (standard JSONL: one JSON per line)
#             try:
#                 obj = json.loads(stripped_line)
#                 data.append(obj)
#                 buffer = []  # Clear buffer since we successfully parsed
#                 continue
#             except json.JSONDecodeError:
#                 # Line is not valid JSON by itself, might be part of multi-line JSON
#                 pass
            
#             # Accumulate line for multi-line JSON parsing
#             buffer.append(line)
#             try:
#                 obj = json.loads("".join(buffer))
#                 data.append(obj)
#                 buffer = []  # Successfully parsed, clear buffer
#             except json.JSONDecodeError:
#                 # Not yet complete, continue accumulating
#                 pass
    
#     # Handle any remaining buffer
#     if buffer:
#         try:
#             obj = json.loads("".join(buffer))
#             data.append(obj)
#         except json.JSONDecodeError:
#             pass
    
#     return Dataset.from_list(data)


# def _safe_load_dataset(data_cfg: DataConfig, sample_size: Optional[int] = None) -> DatasetDict:
#     # Robust fallback: parse pretty-printed JSON objects without relying on pyarrow inference.
#     train_ds = _load_pretty_jsonl(data_cfg.train_file, max_records=sample_size)
#     if data_cfg.eval_file:
#         val_ds = _load_pretty_jsonl(data_cfg.eval_file, max_records=sample_size)
#         return DatasetDict({"train": train_ds, "validation": val_ds})
#     return DatasetDict({"train": train_ds})

def _safe_load_dataset(data_cfg: DataConfig, sample_size: Optional[int] = None) -> DatasetDict:
    dataset_train = load_dataset("json", data_files="data/train_instruct.jsonl")
    dataset_test = load_dataset("json", data_files="data/val_instruct.jsonl")

    return DatasetDict({"train": dataset_train["train"], "validation": dataset_test["train"]})


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

    print("Formatting chat templates...")
    dataset = dataset.map(
        _format_chat,
        remove_columns=dataset["train"].column_names,
        desc="Formatting chat"
    )

    def _tokenize(example):
        return tokenize_with_mask(
            tokenizer=tokenizer,
            text=example["text"],
            max_length=data_cfg.max_length,
            assistant_token=assistant_token,
        )

    print("Tokenizing dataset...")
    dataset = dataset.map(
        _tokenize,
        batched=False,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    return dataset

