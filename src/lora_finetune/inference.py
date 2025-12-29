from typing import List, Dict, Any

from transformers import GenerationConfig

from .config import AppConfig
from .model import load_peft_model
from .tokenization import load_tokenizer, apply_chat_template


def generate_response(cfg: AppConfig, prompt_messages: List[Dict[str, Any]]) -> str:
    tokenizer = load_tokenizer(cfg.tokenizer)
    model = load_peft_model(
        base_model_name=cfg.model.name,
        adapter_dir=cfg.inference.adapter_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    text = apply_chat_template(tokenizer, prompt_messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
    )

    outputs = model.generate(**inputs, generation_config=gen_cfg)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

