"""Reusable Trainer callbacks."""

from __future__ import annotations

import logging
from typing import Optional

from transformers import TrainerCallback, PreTrainedTokenizer

import wandb

LOGGER = logging.getLogger(__name__)


class SamplePreviewCallback(TrainerCallback):  # type: ignore[misc]
    """Generate sample outputs on a cadence during training."""

    def __init__(self, tokenizer: PreTrainedTokenizer, interval: int = 500, use_wandb: bool = False) -> None:
        self._tokenizer = tokenizer
        self._interval = max(interval, 1)
        self._use_wandb = use_wandb

    def on_log(self, args, state, control, **kwargs):  # pragma: no cover - exercised during training
        step = state.global_step
        if step == 0 or step % self._interval != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        prompt = "Reserve"
        model.eval()
        device = next(model.parameters()).device
        import torch

        with torch.no_grad():
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=24,
                do_sample=True,
                temperature=0.9,
            )
        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        LOGGER.info("[step %s] Sample generation: %s", step, text)
        if self._use_wandb:
            wandb.log({"preview/sample": text, "preview/step": step})


__all__ = ["SamplePreviewCallback"]
