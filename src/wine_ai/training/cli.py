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
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training (automatically set by launcher)")
    parser.add_argument("--modal", action="store_true", help="Force execution on Modal cloud (overrides config)")
    parser.add_argument("--local", action="store_true", help="Force local execution (overrides config and environment)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    template = args.prompt_template or PROMPT_TEMPLATE

    # Determine execution mode: local vs Modal
    use_modal = False

    if args.local:
        # Explicit local flag - always use local
        use_modal = False
        print("🏠 Forcing local execution (--local flag)")
    elif args.modal:
        # Explicit Modal flag - always use Modal
        use_modal = True
        print("🌩️  Forcing Modal execution (--modal flag)")
    else:
        # Check config and environment
        from .modal_runner import should_use_modal
        use_modal = should_use_modal(config)

    if use_modal:
        # Execute on Modal
        print("🌩️  Executing training on Modal cloud infrastructure")
        from .modal_runner import execute_remote_training
        execute_remote_training(
            config=config,
            disable_wandb=args.no_wandb,
            prompt_template=template
        )
    else:
        # Execute locally
        print("🏠 Executing training locally")
        train_language_model(config, disable_wandb=args.no_wandb, prompt_template=template)


if __name__ == "__main__":
    main()
