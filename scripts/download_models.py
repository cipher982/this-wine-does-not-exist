#!/usr/bin/env python3
"""Utility to download pretrained checkpoints used by the project."""

from __future__ import annotations

import argparse
import subprocess

DEFAULT_MODELS = [
    "distilgpt2",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Additional model identifier to fetch")
    parser.add_argument("--cache-dir", default="~/.cache/huggingface", help="Model cache directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = DEFAULT_MODELS + (args.model or [])
    for model in models:
        print(f"Downloading {model}...")
        subprocess.run(["uv", "run", "python", "-m", "huggingface_hub.download", "--repo", model, "--local-dir", args.cache_dir], check=False)


if __name__ == "__main__":
    main()
