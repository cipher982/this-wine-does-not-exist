#!/usr/bin/env python3
"""Modal deployment script for wine training."""

import modal
import sys
import os
from pathlib import Path

# Create Modal app
app = modal.App("wine-training-prod")

# Get project root
project_root = Path(__file__).parent

# Build environment dict once
env_vars = {
    "PYTHONPATH": "/app/src",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_CACHE": "/cache/huggingface",
    "TRANSFORMERS_CACHE": "/cache/transformers",
}

# Add WandB key if available
wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    env_vars["WANDB_API_KEY"] = wandb_key

# Create image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install([
        "pandas>=2.2.0",
        "pyarrow>=14.0.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.11.0",
        "wandb>=0.16.0",
        "pydantic>=2.5.0",
        "hf-transfer>=0.1.9",
        "PyYAML",
    ])
    .env(env_vars)
    .workdir("/app")
    .add_local_dir(local_path=project_root / "src", remote_path="/app/src")
    .add_local_dir(local_path=project_root / "configs", remote_path="/app/configs")
)

def train_core(config_path: str, disable_wandb: bool = False):
    """Core training logic."""
    import os, sys, torch

    print(f"🚀 Modal Training Environment:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    from wine_ai.training.configs import load_training_config
    from wine_ai.training.trainers import train_language_model
    from wine_ai.training.cli import PROMPT_TEMPLATE

    config = load_training_config(config_path)
    return train_language_model(config, disable_wandb=disable_wandb, prompt_template=PROMPT_TEMPLATE)

# Individual functions for each GPU - SIMPLE!
@app.function(image=image, gpu="T4", timeout=7200, cpu=4, memory=16384)
def train_T4(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)

@app.function(image=image, gpu="L4", timeout=7200, cpu=4, memory=16384)
def train_L4(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)

@app.function(image=image, gpu="A100", timeout=7200, cpu=4, memory=16384)
def train_A100(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)

@app.function(image=image, gpu="L40S", timeout=7200, cpu=4, memory=16384)
def train_L40S(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)

@app.function(image=image, gpu="H100", timeout=7200, cpu=4, memory=16384)
def train_H100(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)

@app.function(image=image, gpu="H200", timeout=7200, cpu=4, memory=16384)
def train_H200(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)

@app.function(image=image, gpu="B200", timeout=7200, cpu=4, memory=16384)
def train_B200(config_path: str, disable_wandb: bool = False):
    return train_core(config_path, disable_wandb)