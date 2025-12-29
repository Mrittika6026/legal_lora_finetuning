from pathlib import Path
from typing import Optional

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
    )


def run_training(cfg: AppConfig, resume_from_checkpoint: Optional[str] = None):
    tokenizer = load_tokenizer(cfg.tokenizer)
    dataset = load_and_prepare_dataset(
        data_cfg=cfg.data,
        tokenizer=tokenizer,
        assistant_token=cfg.tokenizer.assistant_token,
    )

    model = load_base_model(cfg.model, cfg.quantization)
    model = wrap_with_lora(model, cfg.lora)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

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

    output_dir = Path(cfg.training.output_dir) / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

