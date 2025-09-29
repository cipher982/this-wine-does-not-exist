"""High-level training helpers built on Hugging Face."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from ..data.loaders import load_dataset_with_splits, to_hf_dataset
from .callbacks import SamplePreviewCallback
from .configs import TrainingConfig

LOGGER = logging.getLogger(__name__)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


import wandb


def _prepare_text_column(frame: pd.DataFrame, template: Optional[str]) -> pd.DataFrame:
    frame = frame.copy()
    if template:
        frame["text"] = frame.apply(lambda row: template.format(**row.to_dict()), axis=1)
    else:
        if "description" in frame.columns:
            frame["text"] = frame.apply(
                lambda row: (
                    f"### Instruction:\nWrite a tasting note for the following wine.\n"
                    f"### Input:\nName: {row['name']}\nDescription: {row['description']}\n"
                    "### Response:\n"
                )
                if row.get("description")
                else str(row["name"]),
                axis=1,
            )
        else:
            frame["text"] = frame["name"].astype(str)
    return frame[["text"]]


def train_language_model(
    config: TrainingConfig,
    *,
    disable_wandb: bool = False,
    prompt_template: Optional[str] = None,
) -> None:
    """Train a language model using the processed dataset."""

    # Beautiful training banner
    print("\n" + "=" * 80)
    print("🍷 WINE AI TRAINING PIPELINE")
    print("=" * 80)
    print(f"📋 Job: {config.logging.run_name or 'Wine Description Generation'}")
    print(f"🎯 Model: {config.model.base_model}")
    dataset_source = config.data.dataset_path or "HuggingFace: cipher982/wine-text-126k"
    print(f"📊 Dataset: {dataset_source}")
    print(f"🔢 Max samples: {config.data.max_samples or 'All'}")
    print(f"⚙️  LoRA: {'Enabled' if config.model.use_lora else 'Disabled'}")
    print(f"📈 Epochs: {config.trainer.epochs}")
    print(f"🎲 Seed: {config.seed}")
    print("=" * 80 + "\n")

    set_seed(config.seed)

    # Use HuggingFace dataset by default, fallback to local path if specified
    dataset_bundle = load_dataset_with_splits(config.data.dataset_path if config.data.dataset_path else None)

    train_frame = dataset_bundle.train.copy()
    eval_frame = dataset_bundle.validation.copy()

    if config.data.max_samples is not None:
        train_frame = train_frame.iloc[: config.data.max_samples].reset_index(drop=True)
    if config.data.max_eval_samples is not None and not eval_frame.empty:
        eval_frame = eval_frame.iloc[: config.data.max_eval_samples].reset_index(drop=True)

    print("📝 Preparing training data...")
    train_text = _prepare_text_column(train_frame, prompt_template)
    train_dataset = to_hf_dataset(train_text)

    eval_dataset = None
    if not eval_frame.empty:
        eval_text = _prepare_text_column(eval_frame, prompt_template)
        eval_dataset = to_hf_dataset(eval_text)
        print(f"   Training samples: {len(train_dataset):,}")
        print(f"   Validation samples: {len(eval_dataset):,}")
    else:
        print(f"   Training samples: {len(train_dataset):,}")
        print(f"   No validation data")

    print("🔤 Loading tokenizer...")
    tokenizer_name = config.model.tokenizer or config.model.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad token to EOS token: {tokenizer.pad_token}")

    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(config.model.base_model)

    # Optimize for Apple Silicon
    import torch
    if torch.backends.mps.is_available():
        print("   🚀 Using Metal Performance Shaders (MPS)")
        # Enable memory efficient attention if available
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False  # Disable for training efficiency
    else:
        print("   💻 Using CPU")

    print(f"   Model parameters: {model.num_parameters():,}")

    if config.model.use_lora:
        from peft import LoraConfig, get_peft_model  # type: ignore

        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "c_attn"],
            lora_dropout=config.model.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✨ LoRA enabled: {trainable_params:,} / {total_params:,} parameters trainable ({trainable_params/total_params*100:.1f}%)")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length,
        )

    print("🔤 Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = (
        eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
        if eval_dataset is not None
        else None
    )
    print(f"   Tokenization complete")

    report_to = []
    print("⚙️  Configuring training...")

    if config.logging.use_wandb and not disable_wandb:
        report_to = ["wandb"]
        wandb.init(project=config.logging.project or "wine-ai-dataset", name=config.logging.run_name)
        print(f"   W&B project: {config.logging.project}")

    # Calculate effective batch size
    effective_batch_size = (
        config.trainer.per_device_train_batch_size *
        config.trainer.gradient_accumulation_steps
    )

    training_args = TrainingArguments(
        output_dir=str(config.trainer.output_dir),
        num_train_epochs=config.trainer.epochs,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_steps=config.trainer.warmup_steps,
        eval_strategy=config.trainer.eval_strategy,
        eval_steps=config.trainer.eval_steps,
        save_strategy=config.trainer.save_strategy,
        save_steps=config.trainer.save_steps,
        logging_steps=config.trainer.logging_steps,
        max_steps=config.trainer.max_steps if config.trainer.max_steps is not None else -1,
        report_to=report_to,
        bf16=config.trainer.bf16,
        fp16=config.trainer.fp16,
        save_total_limit=3,
        dataloader_pin_memory=False,  # Disable pin memory to avoid MPS warning
        logging_first_step=False,  # Reduce initial logging noise
    )

    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Learning rate: {config.optimizer.learning_rate}")
    print(f"   Output directory: {config.trainer.output_dir}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    callbacks = [
        SamplePreviewCallback(
            tokenizer=tokenizer,
            interval=config.logging.sample_preview_interval,
            use_wandb=config.logging.use_wandb and not disable_wandb,
        )
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("\n🚀 Starting training...")
    print("=" * 50)
    trainer.train()
    print("\n✅ Training completed!")

    if config.logging.use_wandb and not disable_wandb and getattr(wandb, "run", None) is not None:
        wandb.finish()

    print(f"\n🎯 Model saved to: {config.trainer.output_dir}")
    print("=" * 80)


__all__ = ["train_language_model"]
