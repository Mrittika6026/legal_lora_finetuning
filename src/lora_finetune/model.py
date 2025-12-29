import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .config import ModelConfig, QuantizationConfig, LoraConfigData


def _make_bnb_config(cfg: QuantizationConfig) -> BitsAndBytesConfig:
    dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def load_base_model(model_cfg: ModelConfig, quant_cfg: QuantizationConfig):
    bnb_config = _make_bnb_config(quant_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.trust_remote_code
    )
    if model_cfg.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = model_cfg.use_cache
    return model


def wrap_with_lora(model, lora_cfg: LoraConfigData):
    task_type = TaskType.CAUSAL_LM if lora_cfg.task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=task_type,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_peft_model(base_model_name: str, adapter_dir: str, trust_remote_code: bool = True):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )
    return PeftModel.from_pretrained(base_model, adapter_dir)

