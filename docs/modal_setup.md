# Modal Cloud Training Setup

This guide shows how to run your wine training on Modal's cloud infrastructure with powerful GPUs.

## Quick Setup (2 minutes)

1. **Install Modal support:**
   ```bash
   pip install modal
   # OR with uv:
   uv pip install modal
   ```

2. **Authenticate with Modal:**
   ```bash
   modal login
   ```

3. **Set your WandB key (optional):**
   ```bash
   export WANDB_API_KEY=your_wandb_key_here
   ```

## Usage Options

### Option 1: Command Line Flags
```bash
# Run on Modal cloud
uv run wine-train --config configs/train_description.yaml --modal

# Force local execution
uv run wine-train --config configs/train_description.yaml --local
```

### Option 2: Environment Variable
```bash
# Enable Modal for this session
export USE_MODAL=1
uv run wine-train --config configs/train_description.yaml

# Disable Modal
export USE_MODAL=0
uv run wine-train --config configs/train_description.yaml
```

### Option 3: Configuration File
```yaml
# In configs/train_description.yaml
modal:
  enabled: true        # Change this to enable Modal
  gpu_type: "L40S"     # Choose your GPU
  gpu_count: 1
```

Then run normally:
```bash
uv run wine-train --config configs/train_description.yaml
```

## GPU Options & Pricing

| GPU Type | Memory | Price/hour | Best For |
|----------|--------|------------|----------|
| T4       | 16GB   | ~$0.60     | Testing, small models |
| A100-40GB| 40GB   | ~$2.10     | Medium training jobs |
| A100-80GB| 80GB   | ~$2.50     | Large models, your current job |
| H100     | 80GB   | ~$3.95     | Fastest training |

## Example Configurations

### Development/Testing
```yaml
modal:
  enabled: true
  gpu_type: "T4"
  gpu_count: 1
  timeout: 1800  # 30 minutes
```

### Production Training (Recommended for your job)
```yaml
modal:
  enabled: true
  gpu_type: "A100"
  gpu_count: 1
  timeout: 7200  # 2 hours
  memory: 16     # 16GB RAM
```

### Multi-GPU Training
```yaml
modal:
  enabled: true
  gpu_type: "A100"
  gpu_count: 2     # 2x A100s
  timeout: 10800   # 3 hours
```

## Cost Estimation

Your current 6-hour training job:
- **A100-80GB**: ~$15 total
- **H100**: ~$24 total
- **Much cheaper than hardware issues!**

## Troubleshooting

### "Modal not installed"
```bash
pip install modal
modal login
```

### "Authentication failed"
```bash
modal logout
modal login
```

### "Function timeout"
Increase timeout in config:
```yaml
modal:
  timeout: 14400  # 4 hours
```

## Advanced Features

### Custom Environment
```yaml
modal:
  enabled: true
  gpu_type: "A100"
  image_tag: "custom-v1"  # Use custom Modal image
```

### Multiple Configurations
- `configs/train_local.yaml` - Local development
- `configs/train_modal.yaml` - Cloud training
- `configs/train_modal_fast.yaml` - H100 for speed

## Benefits

✅ **No hardware issues** - bypass your PCIe problems
✅ **Powerful GPUs** - A100s and H100s
✅ **Pay per use** - no idle costs
✅ **Same code** - zero changes to your training logic
✅ **Same commands** - familiar workflow
✅ **Auto-scaling** - handles dependencies automatically