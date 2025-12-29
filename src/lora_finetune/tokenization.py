from typing import Dict, List, Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import TokenizerConfig


def load_tokenizer(cfg: TokenizerConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name,
        trust_remote_code=cfg.trust_remote_code
    )
    tokenizer.padding_side = cfg.padding_side
    if cfg.pad_token == "eos":
        tokenizer.pad_token = tokenizer.eos_token
    elif cfg.pad_token:
        tokenizer.pad_token = cfg.pad_token
    return tokenizer


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, Any]],
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def loss_mask_for_last_assistant(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: List[int],
    assistant_token: str,
) -> List[int]:
    labels = [-100] * len(input_ids)
    assistant_ids = tokenizer.encode(
        assistant_token,
        add_special_tokens=False
    )

    start_idx = None
    for i in range(len(input_ids) - len(assistant_ids) + 1):
        if input_ids[i:i + len(assistant_ids)] == assistant_ids:
            start_idx = i + len(assistant_ids)

    if start_idx is not None and start_idx < len(input_ids):
        labels[start_idx:] = input_ids[start_idx:]
        if tokenizer.eos_token_id is not None and labels[-1] == tokenizer.eos_token_id:
            labels[-1] = -100

    return labels


def tokenize_with_mask(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int,
    assistant_token: str,
) -> Dict[str, List[int]]:
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True
    )
    tokenized["labels"] = loss_mask_for_last_assistant(
        tokenizer=tokenizer,
        input_ids=tokenized["input_ids"],
        assistant_token=assistant_token
    )
    return tokenized

