from pathlib import Path
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .config import AppConfig
from .data import load_and_prepare_dataset
from .model import load_base_model, wrap_with_lora
from .tokenization import load_tokenizer


def build_training_arguments(cfg: AppConfig) -> TrainingArguments:
    train_cfg = cfg.training
    return TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,
        save_steps=train_cfg.save_steps,
        logging_steps=train_cfg.logging_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        optim=train_cfg.optim,
        remove_unused_columns=train_cfg.remove_unused_columns,
        report_to=train_cfg.report_to,
        disable_tqdm=False,  # Explicitly enable progress bars
        dataloader_pin_memory=True,
    )


def run_training(cfg: AppConfig, resume_from_checkpoint: Optional[str] = None):
    print("=" * 60)
    print("Starting LoRA Training")
    print("=" * 60)
    
    # Initialize W&B if configured
    if cfg.wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags or [],
            notes=cfg.wandb.notes,
            config={
                "model": {
                    "name": cfg.model.name,
                    "use_gradient_checkpointing": cfg.model.use_gradient_checkpointing,
                    "use_cache": cfg.model.use_cache,
                },
                "lora": {
                    "r": cfg.lora.r,
                    "lora_alpha": cfg.lora.lora_alpha,
                    "lora_dropout": cfg.lora.lora_dropout,
                    "bias": cfg.lora.bias,
                    "target_modules": cfg.lora.target_modules,
                },
                "data": {
                    "max_length": cfg.data.max_length,
                    "train_file": cfg.data.train_file,
                    "eval_file": cfg.data.eval_file,
                },
                "training": {
                    "num_train_epochs": cfg.training.num_train_epochs,
                    "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
                    "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
                    "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
                    "learning_rate": cfg.training.learning_rate,
                    "warmup_ratio": cfg.training.warmup_ratio,
                    "lr_scheduler_type": cfg.training.lr_scheduler_type,
                    "bf16": cfg.training.bf16,
                    "fp16": cfg.training.fp16,
                    "optim": cfg.training.optim,
                },
                "quantization": {
                    "load_in_4bit": cfg.quantization.load_in_4bit,
                    "bnb_4bit_quant_type": cfg.quantization.bnb_4bit_quant_type,
                    "bnb_4bit_compute_dtype": cfg.quantization.bnb_4bit_compute_dtype,
                },
            }
        )
        print(f"W&B initialized: project={cfg.wandb.project}, name={cfg.wandb.name or 'auto'}")
    
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(cfg.tokenizer)
    
    print("\n[2/4] Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(
        data_cfg=cfg.data,
        tokenizer=tokenizer,
    )
    train_size = len(dataset['train'])
    val_size = len(dataset['validation']) if 'validation' in dataset else 0
    print(f"Train samples: {train_size}")
    if 'validation' in dataset:
        print(f"Val samples: {val_size}")
    
    # Log dataset info to W&B
    if cfg.wandb:
        wandb.config.update({
            "dataset": {
                "train_samples": train_size,
                "val_samples": val_size,
            }
        })

    print("\n[3/4] Loading model and applying LoRA...")
    model = load_base_model(cfg.model, cfg.quantization)
    model = wrap_with_lora(model, cfg.lora)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    print("\n[4/4] Starting training...")
    print(f"Output directory: {cfg.training.output_dir}")
    print(f"Total steps per epoch: ~{len(dataset['train']) // (cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps * 2)}")  # *2 for 2 GPUs
    print("=" * 60)
    
    training_args = build_training_arguments(cfg)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("\n" + "=" * 60)
    print("Training completed! Saving adapter...")
    print("=" * 60)

    output_dir = Path(cfg.training.output_dir) / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Log model artifact to W&B if configured
    if cfg.wandb:
        artifact = wandb.Artifact("model-adapter", type="model")
        artifact.add_dir(str(output_dir))
        wandb.log_artifact(artifact)
        print(f"Model adapter logged to W&B as artifact")
    
    # Finish W&B run
    if cfg.wandb:
        wandb.finish()

