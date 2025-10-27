# HuggingFace Integration Guide

This guide covers how to use Temporal models with the HuggingFace ecosystem, including uploading to and downloading from the HuggingFace Hub.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using HuggingFace Hub](#using-huggingface-hub)
- [Model Sharing](#model-sharing)
- [Advanced Usage](#advanced-usage)

## Installation

Install Temporal with HuggingFace support:

```bash
# Install base package
pip install temporal-forecasting

# Install with HuggingFace support
pip install transformers huggingface-hub
```

Or install all optional dependencies:

```bash
pip install temporal-forecasting[huggingface]
```

## Quick Start

### Create a HuggingFace-Compatible Model

```python
from temporal.hf_interface import create_hf_model

# Create model (uses transformers PreTrainedModel)
model = create_hf_model(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024,
    forecast_horizon=24
)

# Train the model
# ... training code ...

# Save locally
model.save_pretrained("./my-temporal-model")

# Load locally
from temporal.hf_interface import TemporalForForecasting
model = TemporalForForecasting.from_pretrained("./my-temporal-model")
```

### Simple Approach (PyTorchModelHubMixin)

```python
from temporal.hf_interface import TemporalHubMixin

# Create model config
config = {
    'input_dim': 1,
    'd_model': 256,
    'forecast_horizon': 24
}

# Create model
model = TemporalHubMixin(config)

# Train model
# ... training code ...

# Save
model.save_pretrained("./my-temporal-model")

# Load
model = TemporalHubMixin.from_pretrained("./my-temporal-model")
```

## Using HuggingFace Hub

### Prerequisites

1. Create a HuggingFace account at https://huggingface.co/join
2. Get your access token from https://huggingface.co/settings/tokens
3. Login via CLI:

```bash
huggingface-cli login
```

Or in Python:

```python
from huggingface_hub import login
login(token="your_token_here")
```

### Upload Model to Hub

#### Method 1: Using transformers PreTrainedModel

```python
from temporal.hf_interface import TemporalForForecasting, TemporalConfig
import torch

# Create and train model
config = TemporalConfig(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024,
    forecast_horizon=24
)

model = TemporalForForecasting(config)

# Train your model
# ... training code ...

# Push to Hub
model.push_to_hub("your-username/temporal-forecasting-model")
```

#### Method 2: Using PyTorchModelHubMixin

```python
from temporal.hf_interface import TemporalHubMixin

# Create and train model
config = {
    'input_dim': 1,
    'd_model': 256,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'num_heads': 8,
    'd_ff': 1024,
    'forecast_horizon': 24
}

model = TemporalHubMixin(config)

# Train your model
# ... training code ...

# Push to Hub
model.push_to_hub("your-username/temporal-forecasting-model")
```

### Download Model from Hub

#### Download Your Own Model

```python
from temporal.hf_interface import TemporalForForecasting

# Load from Hub
model = TemporalForForecasting.from_pretrained("your-username/temporal-forecasting-model")

# Use for inference
import torch
x = torch.randn(1, 96, 1)
forecast = model.forecast(x)
```

#### Download Community Models

```python
# Load a model shared by the community
model = TemporalForForecasting.from_pretrained("community-user/temporal-model-name")

# Check the model card for expected input format
forecast = model.forecast(x)
```

## Model Sharing

### Create a Model Card

Create a `README.md` in your model directory:

```markdown
---
license: mit
tags:
- time-series
- forecasting
- transformer
- temporal
datasets:
- your-dataset-name
metrics:
- mse
- mae
---

# Temporal Time Series Forecasting Model

## Model Description

This model is a Temporal transformer for time series forecasting.

## Intended Uses

This model is designed for [describe your use case].

## Training Data

The model was trained on [describe your dataset].

## Training Procedure

- Lookback window: 96 time steps
- Forecast horizon: 24 time steps
- Training epochs: 100
- Best validation loss: 0.01

## How to Use

\`\`\`python
from temporal.hf_interface import TemporalForForecasting
import torch

# Load model
model = TemporalForForecasting.from_pretrained("your-username/model-name")

# Prepare your data (96 historical time steps)
x = torch.randn(1, 96, 1)

# Generate forecast (24 future time steps)
forecast = model.forecast(x)
\`\`\`

## Evaluation Results

- MSE: 0.01
- MAE: 0.05
- RMSE: 0.10

## Limitations

[Describe any limitations]

## Citation

\`\`\`bibtex
@software{your_model,
  title = {Your Model Name},
  author = {Your Name},
  year = {2024},
  url = {https://huggingface.co/your-username/model-name}
}
\`\`\`
```

### Upload with Model Card

```python
from temporal.hf_interface import TemporalForForecasting

# Save locally first
model.save_pretrained("./my-model")

# Create README.md in ./my-model/ directory
# Then push everything to Hub

model.push_to_hub(
    "your-username/model-name",
    commit_message="Upload Temporal forecasting model",
    use_auth_token=True
)
```

## Advanced Usage

### Complete Training and Upload Example

```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

from temporal.hf_interface import TemporalForForecasting, TemporalConfig
from temporal.trainer import TimeSeriesDataset, TemporalTrainer

# Prepare data
data = np.random.randn(1000, 1)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Create dataset
train_size = int(0.8 * len(data_normalized))
train_data = data_normalized[:train_size]
val_data = data_normalized[train_size:]

train_dataset = TimeSeriesDataset(train_data, lookback=96, forecast_horizon=24)
val_dataset = TimeSeriesDataset(val_data, lookback=96, forecast_horizon=24)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create HF-compatible model
config = TemporalConfig(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024,
    forecast_horizon=24
)

model = TemporalForForecasting(config)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = TemporalTrainer(
    model=model.temporal,  # Use the underlying Temporal model
    optimizer=optimizer
)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10
)

# Save locally with scaler
model.save_pretrained("./temporal-model")

# Save scaler separately
with open("./temporal-model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save training info
import json
with open("./temporal-model/training_info.json", "w") as f:
    json.dump({
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'best_val_loss': min(history['val_losses'])
    }, f)

# Push to Hub
model.push_to_hub("your-username/temporal-stock-forecasting")

# Also upload scaler and training info
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="./temporal-model/scaler.pkl",
    path_in_repo="scaler.pkl",
    repo_id="your-username/temporal-stock-forecasting"
)
api.upload_file(
    path_or_fileobj="./temporal-model/training_info.json",
    path_in_repo="training_info.json",
    repo_id="your-username/temporal-stock-forecasting"
)
```

### Download and Use with Scaler

```python
from temporal.hf_interface import TemporalForForecasting
from huggingface_hub import hf_hub_download
import pickle
import torch
import numpy as np

# Load model
model = TemporalForForecasting.from_pretrained("your-username/temporal-stock-forecasting")

# Download scaler
scaler_path = hf_hub_download(
    repo_id="your-username/temporal-stock-forecasting",
    filename="scaler.pkl"
)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Use for inference
new_data = np.random.randn(96, 1)
new_data_normalized = scaler.transform(new_data)

x = torch.FloatTensor(new_data_normalized).unsqueeze(0)
forecast = model.forecast(x)

# Denormalize
forecast_np = forecast.cpu().numpy().reshape(-1, 1)
forecast_original = scaler.inverse_transform(forecast_np)

print(f"Forecast: {forecast_original.flatten()}")
```

### Private Models

```python
# Upload as private
model.push_to_hub(
    "your-username/private-model",
    private=True,
    use_auth_token=True
)

# Download private model (requires authentication)
from huggingface_hub import login
login(token="your_token")

model = TemporalForForecasting.from_pretrained(
    "your-username/private-model",
    use_auth_token=True
)
```

### Model Versioning

```python
# Upload specific version
model.push_to_hub(
    "your-username/temporal-model",
    commit_message="Version 1.0.0 - Initial release"
)

# Later, upload new version
model.push_to_hub(
    "your-username/temporal-model",
    commit_message="Version 1.1.0 - Improved accuracy"
)

# Load specific revision
model = TemporalForForecasting.from_pretrained(
    "your-username/temporal-model",
    revision="v1.0.0"  # Git tag or commit hash
)
```

### Inference API

Once uploaded, your model gets a free Inference API endpoint:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/temporal-model"
headers = {"Authorization": f"Bearer {your_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Make prediction
output = query({
    "inputs": [[0.1, 0.2, 0.3, ...]]  # Your time series data
})
```

## Integration with Datasets

### Upload Training Dataset

```python
from datasets import Dataset
import pandas as pd

# Create dataset
df = pd.DataFrame({
    'timestamp': [...],
    'value': [...]
})

dataset = Dataset.from_pandas(df)

# Push to Hub
dataset.push_to_hub("your-username/temporal-training-data")
```

### Reference Dataset in Model Card

```markdown
---
datasets:
- your-username/temporal-training-data
---

This model was trained on [temporal-training-data](https://huggingface.co/datasets/your-username/temporal-training-data).
```

## Best Practices

### 1. Always Include Model Card

Provide clear documentation:
- Model description
- Intended use
- Training data
- Evaluation metrics
- How to use
- Limitations

### 2. Version Your Models

Use Git tags for releases:

```bash
git tag v1.0.0
git push --tags
```

### 3. Include Preprocessing

Upload scaler, normalization params, etc.:

```python
# Save all preprocessing components
import json

preprocessing = {
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_std': scaler.scale_.tolist(),
    'lookback': 96,
    'forecast_horizon': 24
}

with open("./model/preprocessing.json", "w") as f:
    json.dump(preprocessing, f)

# Upload with model
model.push_to_hub("your-username/model")
```

### 4. Add Usage Examples

Include example notebooks:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="./examples/usage_example.ipynb",
    path_in_repo="usage_example.ipynb",
    repo_id="your-username/temporal-model"
)
```

### 5. License Your Models

Add license in model card:

```markdown
---
license: mit
---
```

## Common Issues

### Issue: "Model not found"

**Solution**: Ensure you're authenticated and the model name is correct:

```python
from huggingface_hub import login
login(token="your_token")

model = TemporalForForecasting.from_pretrained(
    "your-username/model-name",
    use_auth_token=True
)
```

### Issue: "Different model architecture"

**Solution**: Config must match when loading:

```python
# Ensure the config in your saved model matches
# Check config.json in the model repository
```

### Issue: "Slow download"

**Solution**: Use local caching:

```python
# Models are cached by default in ~/.cache/huggingface
# To use offline mode:
model = TemporalForForecasting.from_pretrained(
    "your-username/model-name",
    local_files_only=True  # Use cached version
)
```

## Example Models

Here are some example temporal models on HuggingFace Hub:

- `username/temporal-stock-forecasting` - Stock price forecasting
- `username/temporal-energy-demand` - Energy demand prediction
- `username/temporal-weather` - Weather forecasting

## Resources

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Model Hub](https://huggingface.co/models)
- [Datasets Hub](https://huggingface.co/datasets)

## Next Steps

- See `examples/huggingface_example.py` for complete example
- Read `MODEL_PERSISTENCE.md` for general model saving
- Check `TRAINING_GUIDE.md` for training workflows
