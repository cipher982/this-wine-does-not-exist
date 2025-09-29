"""Reusable Trainer callbacks."""

from __future__ import annotations

import logging
import random
from typing import Optional

from transformers import TrainerCallback, PreTrainedTokenizer

import wandb

LOGGER = logging.getLogger(__name__)


class SamplePreviewCallback(TrainerCallback):  # type: ignore[misc]
    """Generate realistic wine sample outputs during training to monitor language evolution."""

    def __init__(self, tokenizer: PreTrainedTokenizer, interval: int = 500, use_wandb: bool = False) -> None:
        self._tokenizer = tokenizer
        self._interval = max(interval, 1)
        self._use_wandb = use_wandb

        # Sample wine prompts for generation testing
        self._sample_prompts = [
            "### Instruction:\nWrite a believable wine tasting description that matches the provided metadata.\n### Input:\nName: Château Reserve Cabernet 2019\nCategory: red\nRegion: napa valley\nPrice: $45.99\n### Response:\n",
            "### Instruction:\nWrite a believable wine tasting description that matches the provided metadata.\n### Input:\nName: Crisp Valley Chardonnay 2021\nCategory: white\nRegion: sonoma\nPrice: $28.50\n### Response:\n",
            "### Instruction:\nWrite a believable wine tasting description that matches the provided metadata.\n### Input:\nName: Domaine Sparkling Rosé 2020\nCategory: sparkling\nRegion: france\nPrice: $65.00\n### Response:\n",
            "### Instruction:\nWrite a believable wine tasting description that matches the provided metadata.\n### Input:\nName: Piedmont Pinot Noir 2018\nCategory: red\nRegion: oregon\nPrice: $32.99\n### Response:\n"
        ]

    def on_log(self, args, state, control, **kwargs):  # pragma: no cover - exercised during training
        step = state.global_step
        if step == 0 or step % self._interval != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return

        # Select random wine prompt for variety
        prompt = random.choice(self._sample_prompts)
        model.eval()
        device = next(model.parameters()).device
        import torch

        with torch.no_grad():
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=80,  # Longer for wine descriptions
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Extract just the generated response (after the prompt)
        generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response_part = generated_text[len(prompt):].strip()

        # Pretty print for console
        wine_name = prompt.split("Name: ")[1].split("\n")[0] if "Name: " in prompt else "Sample Wine"
        print(f"\n🍷 [Step {step}] Generated for {wine_name}:")
        print(f"   {response_part[:100]}{'...' if len(response_part) > 100 else ''}")

        LOGGER.info("[step %s] Generated wine description: %s", step, response_part[:150])

        if self._use_wandb:
            wandb.log({
                "preview/wine_name": wine_name,
                "preview/generated_text": response_part,
                "preview/step": step,
                "preview/char_length": len(response_part)
            })

        model.train()  # Reset to training mode


__all__ = ["SamplePreviewCallback"]