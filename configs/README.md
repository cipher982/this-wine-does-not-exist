# Training Configuration Guide

## Available Configurations

### 🧪 `test_training.yaml`
**Purpose**: Pipeline validation and quick testing
- **Model**: `sshleifer/tiny-gpt2` (11M parameters)
- **Dataset**: 100 train samples, 20 eval samples
- **Training**: 10 steps max, no LoRA
- **Time**: ~30 seconds
- **Use case**: Validate pipeline works before serious training

**Usage:**
```bash
uv run wine-train --config configs/test_training.yaml --no-wandb
```

### 🚀 `train_description.yaml`
**Purpose**: Production wine description generation
- **Model**: `mistralai/Mistral-7B-Instruct-v0.3` (7B parameters)
- **Dataset**: Full 125k wines
- **Training**: 2 epochs with LoRA fine-tuning
- **Time**: ~2-4 hours on GPU
- **Use case**: High-quality wine description generation

**Usage:**
```bash
uv run wine-train --config configs/train_description.yaml
```

## Configuration Structure

### Data Section
```yaml
data:
  dataset_path: data/processed/wine_training_dataset_v1.parquet  # Dataset location
  val_fraction: 0.05           # Validation split percentage
  shuffle_seed: 42             # Reproducibility seed
  max_samples: null            # Limit training samples (null = all)
  max_eval_samples: null       # Limit validation samples (null = all)
```

### Model Section
```yaml
model:
  base_model: mistralai/Mistral-7B-Instruct-v0.3  # HuggingFace model ID
  tokenizer: mistralai/Mistral-7B-Instruct-v0.3   # Tokenizer (usually same)
  max_length: 512              # Maximum sequence length
  use_lora: true               # Enable LoRA fine-tuning
  lora_r: 32                   # LoRA rank (higher = more parameters)
  lora_alpha: 64               # LoRA scaling factor
  lora_dropout: 0.05           # LoRA dropout rate
```

### Training Section
```yaml
trainer:
  output_dir: artifacts/desc_model              # Where to save checkpoints
  epochs: 2                                    # Number of training epochs
  per_device_train_batch_size: 2               # Batch size per GPU
  per_device_eval_batch_size: 2                # Evaluation batch size
  gradient_accumulation_steps: 16              # Effective batch size multiplier
  warmup_steps: 200                           # Learning rate warmup
  eval_strategy: steps                        # When to evaluate (steps/epoch)
  eval_steps: 100                             # Evaluate every N steps
  save_strategy: steps                        # When to save checkpoints
  save_steps: 100                             # Save every N steps
  logging_steps: 20                           # Log every N steps
  max_steps: null                             # Max steps (null = complete epochs)
  bf16: true                                  # Use bfloat16 precision
  fp16: false                                 # Use float16 precision
```

### Logging Section
```yaml
logging:
  use_wandb: true                             # Enable Weights & Biases logging
  project: wine-modernization                 # W&B project name
  entity: null                                # W&B team/user (null = personal)
  run_name: description-generator-dev         # Run identifier
  sample_preview_interval: 400                # Generate samples every N steps
```

## Quick Start

1. **Test the pipeline:**
   ```bash
   uv run wine-train --config configs/test_training.yaml --no-wandb
   ```

2. **Full training run:**
   ```bash
   uv run wine-train --config configs/train_description.yaml
   ```

3. **Custom configuration:**
   - Copy an existing config
   - Modify parameters as needed
   - Run with your custom config

## Model Size Guidelines

| Model | Parameters | VRAM (LoRA) | VRAM (Full) | Training Time |
|-------|------------|-------------|-------------|---------------|
| tiny-gpt2 | 11M | 1GB | 1GB | 30s |
| distilgpt2 | 82M | 2GB | 4GB | 5min |
| gpt2 | 124M | 3GB | 6GB | 10min |
| Mistral-7B | 7B | 12GB | 28GB | 2-4hrs |

## Troubleshooting

- **Out of memory**: Reduce `per_device_train_batch_size` or `max_length`
- **Too slow**: Increase `gradient_accumulation_steps`, reduce `max_samples`
- **Poor quality**: Increase `epochs`, `lora_r`, or use larger model
- **Overfitting**: Add regularization, reduce `epochs` or `lora_r`