# Wine AI Training Guide 🍷

**Complete guide for training wine description generation models on any environment**

---

## 📋 **Project Overview**

This repository contains a modern wine AI training pipeline that fine-tunes language models to generate realistic wine tasting descriptions. The system was recently modernized (September 2025) with a production-ready training infrastructure.

### **Key Components:**
- **Dataset**: 125,787 unique wines with descriptions, prices, regions, categories
- **Model**: GPT-2 fine-tuned for wine description generation using instruction-following format
- **Training Pipeline**: Modern HuggingFace Transformers with beautiful progress output
- **Data Source**: HuggingFace dataset `cipher982/wine-text-126k` (auto-downloads)

---

## 🚀 **Quick Start (Any Environment)**

### **Prerequisites:**
- Python 3.11+
- UV package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- 8GB+ RAM (16GB+ recommended)
- GPU optional but helpful for faster training

### **Setup Commands:**
```bash
# 1. Clone repository
git clone <repository-url>
cd this-wine-does-not-exist

# 2. Install dependencies
uv pip install -e .

# 3. Test pipeline (2 minutes)
uv run wine-train --config configs/test_training.yaml --no-wandb

# 4. Full training run (6+ hours)
uv run wine-train --config configs/train_description.yaml
```

**That's it!** The dataset auto-downloads from HuggingFace on first use.

---

## ⚙️ **Training Configurations**

### **📁 Available Configs:**

#### **`configs/test_training.yaml`** - Pipeline Validation
- **Purpose**: Quick test to verify everything works
- **Model**: `sshleifer/tiny-gpt2` (11M parameters)
- **Data**: 100 training samples, 20 validation
- **Duration**: ~2 minutes
- **Use**: `uv run wine-train --config configs/test_training.yaml --no-wandb`

#### **`configs/train_description.yaml`** - Production Training
- **Purpose**: Full wine description model training
- **Model**: `gpt2` (124M parameters)
- **Data**: Full 125k wine dataset
- **Duration**: ~6-8 hours (varies by hardware)
- **Use**: `uv run wine-train --config configs/train_description.yaml`

### **🎛️ Key Configuration Parameters:**
```yaml
model:
  base_model: gpt2                    # Base model to fine-tune
  max_length: 384                     # Token sequence length
  use_lora: false                     # Full fine-tuning vs LoRA

trainer:
  per_device_train_batch_size: 16     # Batch size per device
  gradient_accumulation_steps: 2      # Effective batch size multiplier
  epochs: 3                           # Training epochs
  learning_rate: 3.0e-05             # Learning rate

data:
  dataset_path: null                  # Uses HF dataset: cipher982/wine-text-126k
  max_samples: null                   # Use all data (or limit for testing)
```

---

## 🍷 **Training Process & Output**

### **What You'll See:**
```
================================================================================
🍷 WINE AI TRAINING PIPELINE
================================================================================
📋 Job: full-wine-training-gpt2
🎯 Model: gpt2
📊 Dataset: HuggingFace: cipher982/wine-text-126k
🔢 Max samples: All
⚙️  LoRA: Disabled
📈 Epochs: 3
🎲 Seed: 99
================================================================================

📦 Loading dataset from HuggingFace: cipher982/wine-text-126k
   ✅ Loaded: 100,629 train, 12,578 validation, 12,580 test
🔤 Loading tokenizer...
   Set pad token to EOS token: <|endoftext|>
🤖 Loading model...
   🚀 Using Metal Performance Shaders (MPS)  # or CUDA/CPU
   Model parameters: 124,439,808
```

### **Periodic Generation Samples:**
Every 200 training steps, you'll see wine generation examples:
```
🍷 [Step 1400] Generated for Château Reserve Cabernet 2019:
   Ruby red in color with garnet highlights. On the nose, this wine is fruity, full-bodied and juicy...
```

This shows you how the model's wine vocabulary evolves during training.

---

## 📊 **Expected Training Metrics**

### **Successful Training Indicators:**
- **Loss trajectory**: Starts ~54 → converges to ~1.8-2.0
- **Generation quality**: Early samples generic → later samples show wine terminology
- **No overfitting**: Eval loss stays close to training loss
- **Stable gradients**: Grad norm ~1.5-3.0 range

### **Platform-Specific Performance:**
| Hardware | Batch Size | Training Time | Memory Usage |
|----------|------------|---------------|--------------|
| M3 Pro Max (64GB) | 16 | 6-8 hours | ~40GB |
| RTX 4090 | 32+ | 2-3 hours | ~20GB VRAM |
| CPU-only | 4-8 | 12+ hours | ~16GB RAM |

---

## 🎯 **Model Architecture & Training Details**

### **Training Format:**
The model learns instruction-following for wine descriptions:
```
### Instruction:
Write a believable wine tasting description that matches the provided metadata.
### Input:
Name: Château Reserve Cabernet 2019
Category: red
Region: napa valley
Price: $45.99
### Response:
[Generated wine description]
```

### **What the Model Learns:**
- **Wine vocabulary**: Tannins, acidity, finish, palate, bouquet, etc.
- **Category-specific language**: Red wine vs white wine vs sparkling descriptions
- **Price-quality correlation**: Language complexity matches wine price point
- **Regional characteristics**: Napa vs Bordeaux vs Tuscany style differences

---

## 🔧 **Hardware Optimization**

### **Apple Silicon (M-series Macs):**
```yaml
# Optimized settings in configs/
trainer:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  bf16: false                    # MPS doesn't support bf16
  fp16: false                    # MPS not recognized as GPU by Accelerate
  dataloader_pin_memory: false   # MPS-specific optimization
```

### **NVIDIA GPUs:**
```yaml
# For RTX 4090 or similar
trainer:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  bf16: true                     # Better than fp16 on modern GPUs
  fp16: false
```

### **CPU-Only Systems:**
```yaml
# For servers without GPU
trainer:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  bf16: false
  fp16: false
```

---

## 📁 **Repository Structure**

```
├── src/wine_ai/              # Core Python package
│   ├── data/loaders.py        # Dataset loading (HF + local fallback)
│   ├── training/              # Training pipeline
│   │   ├── cli.py            # Command-line interface
│   │   ├── trainers.py       # Main training logic
│   │   ├── callbacks.py      # Sample generation during training
│   │   └── configs.py        # Configuration handling
│   └── models/               # Model architectures and inference
├── configs/                  # Training configurations
│   ├── test_training.yaml    # Quick validation (2 min)
│   └── train_description.yaml # Full training (6+ hours)
├── notebooks/                # Jupyter notebooks for exploration
└── artifacts/                # Training outputs (gitignored)
```

---

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **Out of Memory:**
```bash
# Reduce batch size in config
per_device_train_batch_size: 8  # Instead of 16
gradient_accumulation_steps: 4   # Keep effective batch size
```

#### **HuggingFace Dataset Not Found:**
- **Cause**: Dataset still being uploaded/processed
- **Solution**: Automatic fallback to local files with clear error message
- **Manual**: Download files per `data/DOWNLOAD.md` if needed

#### **Slow Training:**
```bash
# Check hardware utilization
htop                    # CPU usage
nvidia-smi             # GPU usage (NVIDIA)
Activity Monitor       # macOS GPU usage
```

#### **Generation Quality Poor:**
- **Early in training**: Normal, quality improves by epoch 1-2
- **After completion**: Try longer training or larger model

### **Platform-Specific Notes:**

#### **macOS (Apple Silicon):**
- Uses Metal Performance Shaders (MPS) automatically
- Unified memory allows larger batch sizes than traditional GPU setups
- FP16 not supported - uses FP32 (still fast)

#### **Linux + NVIDIA:**
- CUDA automatically detected and used
- Enable BF16 for Ampere+ GPUs (RTX 30/40 series)
- Monitor VRAM with `nvidia-smi`

#### **CPU-Only:**
- Still works, just slower
- Uses all available CPU cores automatically
- Consider cloud GPU instances for faster iteration

---

## 📈 **Training Outputs**

### **During Training:**
- **Checkpoints**: `artifacts/[model_name]/checkpoint-N/`
- **Logs**: Console output + W&B dashboard (if enabled)
- **Samples**: Periodic wine descriptions showing learning progress

### **After Training:**
- **Final model**: `artifacts/[model_name]/` (ready for inference)
- **Training metrics**: Loss curves, learning rates, sample outputs
- **Model size**: ~500MB for GPT-2 checkpoint

### **Using Trained Model:**
```python
from wine_ai.models.text_generation import WineGPT, WineGPTConfig

config = WineGPTConfig(model_name="artifacts/full_wine_model")
generator = WineGPT(config)

# Generate wine description
prompt = """### Instruction:
Write a believable wine tasting description that matches the provided metadata.
### Input:
Name: Test Cabernet 2020
Category: red
Region: napa valley
Price: $45.99
### Response:
"""

description = generator.generate(prompt)
print(description)
```

---

## 🎯 **Recent Improvements (Sept 2025)**

### **Code Quality:**
- ✅ Removed dangerous ImportError fallbacks
- ✅ Clean imports with proper error handling
- ✅ Modern HuggingFace Transformers integration

### **Training Experience:**
- ✅ Beautiful progress output with job parameters
- ✅ Periodic wine generation samples during training
- ✅ Apple Silicon optimization (MPS backend)
- ✅ Professional W&B logging integration

### **Data Management:**
- ✅ HuggingFace Datasets integration (auto-download)
- ✅ Fallback to local files when needed
- ✅ Image linking preserved for multimodal training
- ✅ Git repository stays lightweight (code only)

---

## 📞 **Support & Next Steps**

### **If Training Fails:**
1. **Check hardware requirements** (8GB+ RAM minimum)
2. **Verify dataset access** (HF dataset or local files)
3. **Try test config first**: `configs/test_training.yaml`
4. **Check logs** for specific error messages

### **For Better Results:**
1. **Use larger models**: Switch to `microsoft/DialoGPT-large` in config
2. **Enable LoRA**: Set `use_lora: true` for parameter-efficient training
3. **Longer training**: Increase epochs or remove `max_steps` limit
4. **Custom prompts**: Modify template in `src/wine_ai/training/cli.py`

### **Scaling Up:**
- **Multi-GPU**: Configure `CUDA_VISIBLE_DEVICES` and adjust batch sizes
- **Cloud Training**: AWS/GCP instances with GPU support
- **Distributed**: Use HuggingFace Accelerate for multi-node training

---

**🍷 Ready to create world-class wine descriptions with AI! Good luck with your training!**