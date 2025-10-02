"""Typed configuration schema for training routines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field



class DataSection(BaseModel):
    dataset_path: Optional[Path] = Field(default=None, description="Override training dataset path")
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    shuffle_seed: int = 42
    max_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class ModelSection(BaseModel):
    base_model: str = Field(default="distilgpt2")
    tokenizer: Optional[str] = None
    max_length: int = 128
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class OptimizerSection(BaseModel):
    learning_rate: float = 5e-5
    weight_decay: float = 0.0


class TrainerSection(BaseModel):
    output_dir: Path = Path("artifacts/models")
    epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    eval_strategy: str = Field(default="steps")
    eval_steps: int = 200
    save_strategy: str = Field(default="steps")
    save_steps: int = 200
    logging_steps: int = 50
    max_steps: Optional[int] = None
    bf16: bool = True
    fp16: bool = False


class LoggingSection(BaseModel):
    use_wandb: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    sample_preview_interval: int = 500


class ModalSection(BaseModel):
    """Configuration for Modal cloud execution."""
    enabled: bool = Field(default=False, description="Enable Modal cloud execution")
    gpu_type: str = Field(default="A100", description="GPU type: T4, A100, L4, H100")
    gpu_count: int = Field(default=1, ge=1, le=8, description="Number of GPUs")
    timeout: int = Field(default=7200, description="Function timeout in seconds")
    cpu_count: int = Field(default=4.0, description="Number of CPU cores")
    memory: int = Field(default=16, description="Memory in GB")
    image_tag: Optional[str] = Field(default=None, description="Custom Modal image tag")


class TrainingConfig(BaseModel):
    seed: int = 42
    data: DataSection = DataSection()
    model: ModelSection = ModelSection()
    optimizer: OptimizerSection = OptimizerSection()
    trainer: TrainerSection = TrainerSection()
    logging: LoggingSection = LoggingSection()
    modal: ModalSection = ModalSection()


DEFAULT_CONFIG_PATH = Path("configs/default_training.yaml")


def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    if path is None:
        if DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
        else:
            return TrainingConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return TrainingConfig.model_validate(payload)


__all__ = ["TrainingConfig", "load_training_config"]
