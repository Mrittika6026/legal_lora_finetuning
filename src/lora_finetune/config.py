from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict

import yaml


@dataclass
class ModelConfig:
    name: str
    trust_remote_code: bool = True
    use_gradient_checkpointing: bool = True
    use_cache: bool = False


@dataclass
class TokenizerConfig:
    name: str
    trust_remote_code: bool = True
    padding_side: str = "right"
    pad_token: str = "eos"
    assistant_token: str = "<|assistant|>"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfigData:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None


@dataclass
class DataConfig:
    train_file: str
    eval_file: Optional[str] = None
    max_length: int = 4096


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    optim: str = "adamw_torch"
    remove_unused_columns: bool = False
    report_to: str = "none"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 300
    temperature: float = 0.2


@dataclass
class InferenceConfig:
    adapter_dir: str
    system_prompt: str = "You are a helpful legal assistant."


@dataclass
class WandbConfig:
    project: str
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class AppConfig:
    model: ModelConfig
    tokenizer: TokenizerConfig
    quantization: QuantizationConfig
    lora: LoraConfigData
    data: DataConfig
    training: TrainingConfig
    generation: GenerationConfig
    inference: InferenceConfig
    wandb: Optional[WandbConfig] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_dataclass(cls, data: Dict[str, Any]):
    return cls(**data)


def load_config(path: str) -> AppConfig:
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)
    wandb_config = None
    if "wandb" in raw and raw["wandb"]:
        wandb_config = _as_dataclass(WandbConfig, raw["wandb"])
    return AppConfig(
        model=_as_dataclass(ModelConfig, raw["model"]),
        tokenizer=_as_dataclass(TokenizerConfig, raw["tokenizer"]),
        quantization=_as_dataclass(QuantizationConfig, raw["quantization"]),
        lora=_as_dataclass(LoraConfigData, raw["lora"]),
        data=_as_dataclass(DataConfig, raw["data"]),
        training=_as_dataclass(TrainingConfig, raw["training"]),
        generation=_as_dataclass(GenerationConfig, raw.get("generation", {})),
        inference=_as_dataclass(InferenceConfig, raw["inference"]),
        wandb=wandb_config,
    )

