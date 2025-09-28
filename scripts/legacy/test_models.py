#!/usr/bin/env python3
"""
Test script to verify all models and data are working
Run this to ensure everything loads before building the real-time pipeline
"""

import os
import sys
import pickle
import gzip
import json
from pathlib import Path

# Test results tracking
results = {
    "data_loading": False,
    "lstm_model": False,
    "gpt2_model": False,
    "stylegan2_model": False,
    "tensorflow": False,
    "pytorch": False,
    "cuda_available": False
}

print("=" * 60)
print("WINE MODEL VERIFICATION SCRIPT")
print("=" * 60)

# 1. Test Python environment
print("\n1. PYTHON ENVIRONMENT")
print("-" * 40)
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

# 2. Test ML frameworks
print("\n2. ML FRAMEWORKS")
print("-" * 40)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    if sys.platform == "darwin":  # macOS
        print(f"   Metal GPU: {len(tf.config.list_physical_devices('GPU'))} devices")
    results["tensorflow"] = True
except ImportError as e:
    print(f"❌ TensorFlow not installed: {e}")
except Exception as e:
    print(f"⚠️  TensorFlow issue: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        results["cuda_available"] = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"   Apple Metal GPU available")
        results["cuda_available"] = True  # Count Metal as GPU
    else:
        print("   CPU only")
    results["pytorch"] = True
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
except Exception as e:
    print(f"⚠️  PyTorch issue: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers not installed: {e}")

# 3. Test data loading
print("\n3. DATA LOADING")
print("-" * 40)

data_path = Path("data/00_SOURCE/wine_scraped_125k.pickle.gz")
if data_path.exists():
    try:
        with gzip.open(data_path, 'rb') as f:
            wines = pickle.load(f)
        print(f"✅ Loaded {len(wines):,} wines from pickle")
        print(f"   Columns: {wines.columns.tolist()}")
        print(f"   Sample wine: {wines.iloc[0]['name']}")
        results["data_loading"] = True
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
else:
    print(f"❌ Data file not found: {data_path}")

# 4. Test LSTM name model
print("\n4. LSTM NAME MODEL")
print("-" * 40)

model_path = Path("data/03_MODELS/model_weights_name.h5")
if model_path.exists() and results["tensorflow"]:
    try:
        import tensorflow as tf
        # Try to load just weights first
        print(f"   Found model file: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Load model architecture from JSON
        json_path = Path("data/03_MODELS/model_char_DESCS.json")
        if json_path.exists():
            with open(json_path) as f:
                model_config = json.load(f)
            print(f"   Architecture: {model_config.get('class_name', 'Unknown')}")

        # Note: Full model loading would require the architecture
        print("⚠️  Model file exists but needs architecture to load")
        print("   Will need to rebuild model architecture from notebooks")

    except Exception as e:
        print(f"❌ Failed to check LSTM model: {e}")
else:
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
    else:
        print("❌ TensorFlow not available")

# 5. Test GPT-2 model
print("\n5. GPT-2 DESCRIPTION MODEL")
print("-" * 40)

gpt2_paths = [
    Path("gpt2_deepspeed/gpt2-xl_model"),
    Path("data/03_MODELS/model_description_weights.h5")
]

for path in gpt2_paths:
    if path.exists():
        print(f"   Found: {path}")
        if path.is_dir():
            files = list(path.glob("*"))
            print(f"   Contains {len(files)} files")

if results["pytorch"]:
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("✅ Can import GPT2 classes")
        # Note: Actually loading the model would require the full model files
    except Exception as e:
        print(f"⚠️  GPT2 import issue: {e}")

# 6. Test StyleGAN2 setup
print("\n6. STYLEGAN2 LABEL MODEL")
print("-" * 40)

# Look for StyleGAN2 pickle files
stylegan_files = list(Path(".").glob("**/*.pkl"))
if stylegan_files:
    print(f"   Found {len(stylegan_files)} .pkl files")
    for f in stylegan_files[:3]:  # Show first 3
        print(f"   - {f}")
else:
    print("❌ No StyleGAN2 .pkl files found")
    print("   Will need to download or retrain")

# 7. Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

working = []
missing = []
issues = []

if results["data_loading"]:
    working.append("✅ Wine dataset (125k wines)")
else:
    missing.append("❌ Wine dataset")

if results["tensorflow"]:
    working.append("✅ TensorFlow installed")
else:
    missing.append("❌ TensorFlow")

if results["pytorch"]:
    working.append("✅ PyTorch installed")
else:
    missing.append("❌ PyTorch")

if results["cuda_available"]:
    working.append("✅ GPU acceleration available")
else:
    issues.append("⚠️  No GPU acceleration")

if working:
    print("\nWORKING:")
    for item in working:
        print(f"  {item}")

if missing:
    print("\nMISSING:")
    for item in missing:
        print(f"  {item}")

if issues:
    print("\nISSUES:")
    for item in issues:
        print(f"  {item}")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("1. Install missing dependencies: pip install -r requirements-2025.txt")
print("2. Rebuild LSTM model architecture from notebooks")
print("3. Load or retrain GPT-2 model")
print("4. Find or regenerate StyleGAN2 labels")
print("=" * 60)