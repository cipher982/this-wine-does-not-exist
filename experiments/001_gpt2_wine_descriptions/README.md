# Experiment 001 – GPT-2 Wine Descriptions

This experiment contains the original 2020–2021 DeepSpeed fine-tuning harness for GPT-2 XL.
It has been preserved for historical reference and will be replaced by the modern training
utilities in `src/wine_ai/training`.

## Files
- `finetune.py` – original DeepSpeed script
- `ds_config_1gpu.json` – runtime configuration
- `tokenizer_gpt2/` – tokenizer artifacts with custom tokens

## Migration Notes
- Prefer the new Hugging Face Trainer pipeline (`wine-train` CLI) for future runs.
- Use the Parquet dataset outputs under `data/processed/` instead of the legacy TXT files.
