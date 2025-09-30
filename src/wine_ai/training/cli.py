"""Command line interface for training Wine AI models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .configs import load_training_config
from .trainers import train_language_model

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Write a believable wine tasting description that matches the provided metadata.\n"
    "### Input:\n"
    "Name: {name}\n"
    "Category: {category}\n"
    "Region: {region}\n"
    "Price: ${price:.2f}\n"
    "### Response:\n{description}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_description.yaml"), help="Training configuration YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--prompt-template", type=str, default=None, help="Optional Python format string template")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    template = args.prompt_template or PROMPT_TEMPLATE
    train_language_model(config, disable_wandb=args.no_wandb, prompt_template=template)


if __name__ == "__main__":
    main()
