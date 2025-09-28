# Wine Generation: From Static to Dynamic

## The Goal: Keep the Classic AI, Enable Real-Time Generation

### What We're Keeping (The Time Capsule)
- **LSTM for Names**: Your 2020-era character-level LSTM
- **GPT-2 for Descriptions**: The 1.5B parameter model you fine-tuned
- **StyleGAN2 for Labels**: If we can find/retrain it

### What We're Upgrading
- **Static Files → Real-Time API**: Generate on each page visit
- **No Options → User Control**: Wine type, sweetness, price range sliders
- **Random Selection → Seeded Generation**: Shareable wine URLs

## Architecture Design

### Option 1: Lightweight Real-Time (Recommended)
```python
# Fast API backend running your original models
from fastapi import FastAPI
import tensorflow as tf
import torch

app = FastAPI()

# Load once at startup
name_model = tf.keras.models.load_model('model_weights_name.h5')
desc_model = load_gpt2_model()  # Your fine-tuned GPT-2

@app.get("/generate")
async def generate_wine(
    wine_type: str = None,
    sweetness: float = 0.5,  # 0-1 scale
    price_range: str = "medium",
    seed: int = None
):
    # Generate name with LSTM
    name = generate_name_lstm(name_model, wine_type, seed)

    # Generate description with GPT-2
    description = generate_desc_gpt2(desc_model, name, sweetness)

    # Generate label (either cached StyleGAN2 or placeholder)
    label_url = get_or_generate_label(name, wine_type)

    return {
        "name": name,
        "description": description,
        "label": label_url,
        "permalink": f"/wine/{seed}"  # Shareable!
    }
```

### Option 2: Hybrid Approach
- **Names & Descriptions**: Real-time (fast enough)
- **Labels**: Pre-generated pool of 10,000 StyleGAN2 images
  - Match label to wine type/name hash
  - Still unique feeling but no GPU needed in production

### Option 3: Full Classic Stack
- Run all three models in real-time
- Requires GPU server (expensive)
- Most authentic to original vision

## Implementation Path

### Step 1: Get Models Running Locally
```bash
# Test LSTM name generation
python generate_wine_name.py --model data/03_MODELS/model_weights_name.h5

# Test GPT-2 description generation
python generate_description.py --model gpt2_deepspeed/gpt2-xl_model

# Check for StyleGAN2 model
find . -name "*.pkl" -size +100M  # Look for model pickle
```

### Step 2: Build Generation API
```python
# api.py
import numpy as np
from typing import Optional

class WineGenerator:
    def __init__(self):
        self.name_model = self.load_lstm_model()
        self.desc_model = self.load_gpt2_model()
        self.wine_types = ['Cabernet', 'Chardonnay', 'Pinot Noir', ...]

    def generate(self, seed: Optional[int] = None, **params):
        if seed:
            np.random.seed(seed)

        # Your original generation logic
        name = self.generate_name(params.get('wine_type'))
        desc = self.generate_description(name, params)

        return Wine(name, desc, self.estimate_price(name, desc))
```

### Step 3: Interactive Features
```javascript
// Frontend wine customization
<div class="wine-controls">
  <select id="wine-type">
    <option>Random</option>
    <option>Cabernet Sauvignon</option>
    <option>Chardonnay</option>
    <option>Pinot Noir</option>
  </select>

  <input type="range" id="sweetness" min="0" max="100">
  <label>Sweetness</label>

  <input type="range" id="price" min="0" max="500">
  <label>Price Range</label>

  <button onclick="generateWine()">Generate My Wine</button>
</div>
```

## Performance Targets

### With Original Models (2020 tech)
- **LSTM Name**: ~50ms per generation
- **GPT-2 Description**: ~500ms per generation
- **StyleGAN2 Label**: ~2-3 seconds (needs GPU)

### On Modern Hardware (2025)
- **CPU inference**: 5-10x faster than 2020
- **Caching**: Redis for recent generations
- **CDN**: Cloudflare for static assets

## Deployment Options

### Vercel/Netlify + Serverless
```python
# api/generate.py (Vercel serverless)
def handler(request):
    # Lightweight models only
    # LSTM works great serverless
    # GPT-2 might be too big
    pass
```

### Docker on Your Cube Server
```dockerfile
FROM python:3.11
RUN pip install tensorflow transformers fastapi
COPY models/ /app/models/
COPY api.py /app/
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
```

### Hosted GPU Service (for StyleGAN2)
- Replicate.com API for image generation
- Modal.com for full stack
- Your Cube server with RTX 3090

## Database Schema (Optional)
```sql
-- Store generated wines for sharing/rating
CREATE TABLE wines (
    id SERIAL PRIMARY KEY,
    seed INTEGER UNIQUE,
    name VARCHAR(200),
    description TEXT,
    wine_type VARCHAR(50),
    parameters JSONB,
    label_url VARCHAR(500),
    generated_at TIMESTAMP,
    views INTEGER DEFAULT 0,
    rating FLOAT
);

-- User favorites
CREATE TABLE favorites (
    user_id VARCHAR(100),
    wine_id INTEGER REFERENCES wines(id),
    created_at TIMESTAMP
);
```

## The Fun Stuff: Seeded Generation

```python
def wine_from_text(text: str) -> Wine:
    """Generate consistent wine from any text"""
    seed = hash(text) % 2**32
    return generate_wine(seed=seed)

# Examples:
wine_from_text("OpenAI")     # Always generates same wine
wine_from_text("Claude")     # Different but consistent
wine_from_text("2024-12-24") # Date-based wine of the day
```

## MVP in 3 Steps

1. **Get LSTM name generation working** (1 hour)
   - Load model, generate names
   - Wrap in simple API

2. **Add GPT-2 descriptions** (2 hours)
   - Load fine-tuned model
   - Chain with name generation

3. **Deploy to web** (2 hours)
   - FastAPI backend
   - Simple HTML/JS frontend
   - Host on your Cube server

This preserves the "time capsule" nature while making it interactive and modern!