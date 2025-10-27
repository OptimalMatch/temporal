# Model Persistence Guide

This guide covers how to save trained models and reload them for inference without retraining.

## Table of Contents
- [Quick Start](#quick-start)
- [Saving Models](#saving-models)
- [Loading Models](#loading-models)
- [Model Checkpointing](#model-checkpointing)
- [Production Deployment](#production-deployment)
- [Best Practices](#best-practices)

## Quick Start

### Save a Trained Model

```python
from temporal import Temporal
import torch

# After training
model = Temporal(input_dim=1, forecast_horizon=24)
# ... training code ...

# Save model weights
torch.save(model.state_dict(), 'temporal_model.pt')

# Save entire model (alternative)
torch.save(model, 'temporal_model_full.pt')
```

### Load and Use a Saved Model

```python
from temporal import Temporal
import torch

# Load model
model = Temporal(input_dim=1, forecast_horizon=24)
model.load_state_dict(torch.load('temporal_model.pt'))
model.eval()  # Set to evaluation mode

# Make predictions
x = torch.randn(1, 96, 1)  # Example input
forecast = model.forecast(x)
```

## Saving Models

### Method 1: Save State Dict (Recommended)

Save only the model parameters (weights and biases):

```python
import torch
from temporal import Temporal

# Train your model
model = Temporal(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024,
    forecast_horizon=24
)

# ... training code ...

# Save state dict
torch.save(model.state_dict(), 'temporal_weights.pt')
```

**Advantages**:
- Smaller file size
- More flexible (can load into different model structures)
- Recommended by PyTorch

### Method 2: Save Complete Model

Save the entire model object:

```python
# Save complete model
torch.save(model, 'temporal_full.pt')
```

**Advantages**:
- Simpler to save
- Includes model architecture

**Disadvantages**:
- Larger file size
- Less flexible
- Requires same code structure when loading

### Method 3: Save with Metadata

Save model with configuration and training info:

```python
import torch
import json

# Model configuration
config = {
    'input_dim': 1,
    'd_model': 256,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'num_heads': 8,
    'd_ff': 1024,
    'forecast_horizon': 24,
    'dropout': 0.1
}

# Save model and config
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'epoch': epoch,
    'train_loss': train_loss,
    'val_loss': val_loss
}

torch.save(checkpoint, 'temporal_checkpoint.pt')

# Also save config as JSON for easy reference
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=4)
```

### Method 4: Save with Optimizer State

For resuming training:

```python
# Save everything needed to resume training
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'config': config
}

torch.save(checkpoint, 'temporal_training_checkpoint.pt')
```

### Method 5: Save with Scaler (for Denormalization)

Include data preprocessing information:

```python
from sklearn.preprocessing import StandardScaler
import pickle

# Train model with normalized data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# ... training ...

# Save model
torch.save(model.state_dict(), 'temporal_weights.pt')

# Save scaler separately
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Or save together
checkpoint = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'config': config
}
torch.save(checkpoint, 'temporal_complete.pt')
```

## Loading Models

### Load State Dict

```python
from temporal import Temporal
import torch

# Recreate model with same architecture
model = Temporal(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024,
    forecast_horizon=24
)

# Load weights
model.load_state_dict(torch.load('temporal_weights.pt'))

# Set to evaluation mode
model.eval()

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Load from Checkpoint with Config

```python
import torch
from temporal import Temporal

# Load checkpoint
checkpoint = torch.load('temporal_checkpoint.pt')

# Extract config
config = checkpoint['config']

# Create model from config
model = Temporal(**config)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model trained for {checkpoint['epoch']} epochs")
print(f"Training loss: {checkpoint['train_loss']:.6f}")
print(f"Validation loss: {checkpoint['val_loss']:.6f}")
```

### Load Full Model

```python
import torch

# Load complete model
model = torch.load('temporal_full.pt')
model.eval()
```

### Load for CPU When Trained on GPU

```python
# Load model trained on GPU to CPU
model.load_state_dict(torch.load('temporal_weights.pt', map_location=torch.device('cpu')))
model.eval()
```

### Resume Training from Checkpoint

```python
import torch
from temporal import Temporal
from temporal.trainer import TemporalTrainer

# Load checkpoint
checkpoint = torch.load('temporal_training_checkpoint.pt')

# Recreate model
model = Temporal(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Recreate optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Resume training
start_epoch = checkpoint['epoch'] + 1

trainer = TemporalTrainer(model=model, optimizer=optimizer)

# Continue training
for epoch in range(start_epoch, num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    # ...
```

## Model Checkpointing

### Save Best Model During Training

```python
from temporal.trainer import TemporalTrainer
import torch

# Create trainer
trainer = TemporalTrainer(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.MSELoss()
)

# Train with automatic best model saving
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    early_stopping_patience=10,
    save_path="best_model.pt"  # Automatically saves best model
)
```

### Custom Checkpoint Callback

```python
import torch
import os

class CheckpointCallback:
    def __init__(self, save_dir='checkpoints', save_every=10):
        self.save_dir = save_dir
        self.save_every = save_every
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, model, optimizer, train_loss, val_loss):
        # Save every N epochs
        if (epoch + 1) % self.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")

# Usage in training loop
callback = CheckpointCallback(save_dir='checkpoints', save_every=10)

for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)

    callback.on_epoch_end(epoch, model, optimizer, train_loss, val_loss)
```

## Production Deployment

### Create a Reusable Model Class

```python
import torch
from temporal import Temporal
import pickle
import json
import numpy as np

class TemporalPredictor:
    """
    Production-ready model wrapper for inference.
    """
    def __init__(self, model_path, config_path=None, scaler_path=None):
        """
        Load trained model for inference.

        Args:
            model_path: Path to saved model weights
            config_path: Path to model config JSON
            scaler_path: Path to saved scaler pickle
        """
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Try to load from checkpoint
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                raise ValueError("Config required. Provide config_path or save config in checkpoint.")

        # Create model
        self.model = Temporal(**self.config)

        # Load weights
        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        # Load scaler if provided
        self.scaler = None
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def predict(self, data, denormalize=True):
        """
        Generate forecast.

        Args:
            data: Input time series (numpy array or tensor)
            denormalize: Whether to denormalize predictions

        Returns:
            Forecast as numpy array
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)

        # Add batch dimension if needed
        if data.dim() == 2:
            data = data.unsqueeze(0)

        # Move to device
        data = data.to(self.device)

        # Generate forecast
        with torch.no_grad():
            forecast = self.model.forecast(data)

        # Convert to numpy
        forecast = forecast.cpu().numpy()

        # Denormalize if scaler provided
        if denormalize and self.scaler is not None:
            original_shape = forecast.shape
            forecast = forecast.reshape(-1, forecast.shape[-1])
            forecast = self.scaler.inverse_transform(forecast)
            forecast = forecast.reshape(original_shape)

        return forecast

    def save_complete(self, save_dir):
        """Save model, config, and scaler together."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        # Save scaler if exists
        if self.scaler:
            with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)

# Usage
predictor = TemporalPredictor(
    model_path='temporal_checkpoint.pt',
    config_path='model_config.json',
    scaler_path='scaler.pkl'
)

# Make predictions
data = np.random.randn(96, 1)  # Historical data
forecast = predictor.predict(data)
print(f"Forecast shape: {forecast.shape}")
```

### Save as Single Package

```python
import torch
import json
import pickle

def save_model_package(model, config, scaler, path):
    """
    Save model, config, and scaler in a single file.

    Args:
        model: Trained Temporal model
        config: Model configuration dict
        scaler: Fitted data scaler
        path: Save path
    """
    package = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'scaler': scaler
    }
    torch.save(package, path)
    print(f"Model package saved to {path}")

def load_model_package(path):
    """
    Load complete model package.

    Args:
        path: Path to model package

    Returns:
        model, config, scaler
    """
    from temporal import Temporal

    package = torch.load(path)

    # Recreate model
    model = Temporal(**package['config'])
    model.load_state_dict(package['model_state_dict'])
    model.eval()

    return model, package['config'], package['scaler']

# Save
save_model_package(model, config, scaler, 'temporal_package.pt')

# Load
model, config, scaler = load_model_package('temporal_package.pt')
```

## Best Practices

### 1. Always Save Config with Model

```python
# DON'T: Just save weights
torch.save(model.state_dict(), 'model.pt')

# DO: Save weights + config
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'input_dim': 1,
        'd_model': 256,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'num_heads': 8,
        'd_ff': 1024,
        'forecast_horizon': 24
    }
}
torch.save(checkpoint, 'model.pt')
```

### 2. Version Your Models

```python
import datetime

# Add version and timestamp
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'version': '1.0.0',
    'timestamp': datetime.datetime.now().isoformat(),
    'pytorch_version': torch.__version__
}

# Save with versioned name
version = '1.0.0'
torch.save(checkpoint, f'temporal_v{version}.pt')
```

### 3. Validate Loaded Models

```python
def validate_model_load(model, test_data):
    """
    Verify model loads correctly by testing on sample data.
    """
    try:
        model.eval()
        with torch.no_grad():
            output = model.forecast(test_data)
        print("✓ Model loaded successfully")
        print(f"  Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        return False

# After loading
test_input = torch.randn(1, 96, 1)
validate_model_load(model, test_input)
```

### 4. Save Training History

```python
import json

# Save training metrics
training_info = {
    'train_losses': history['train_losses'],
    'val_losses': history['val_losses'],
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'total_epochs': num_epochs
}

with open('training_history.json', 'w') as f:
    json.dump(training_info, f, indent=4)
```

### 5. Model Versioning Structure

```
models/
├── v1.0.0/
│   ├── model.pt
│   ├── config.json
│   ├── scaler.pkl
│   └── training_history.json
├── v1.1.0/
│   ├── model.pt
│   ├── config.json
│   ├── scaler.pkl
│   └── training_history.json
└── latest -> v1.1.0/
```

## Example: Complete Save/Load Workflow

```python
import torch
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer
from torch.utils.data import DataLoader

# ==================== TRAINING ====================

# Prepare data
data = np.random.randn(1000, 1)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Split data
train_size = int(0.8 * len(data_normalized))
train_data = data_normalized[:train_size]
val_data = data_normalized[train_size:]

# Create datasets
train_dataset = TimeSeriesDataset(train_data, lookback=96, forecast_horizon=24)
val_dataset = TimeSeriesDataset(val_data, lookback=96, forecast_horizon=24)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model config
config = {
    'input_dim': 1,
    'd_model': 256,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'num_heads': 8,
    'd_ff': 1024,
    'forecast_horizon': 24,
    'dropout': 0.1
}

# Create and train model
model = Temporal(**config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = TemporalTrainer(model=model, optimizer=optimizer)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10,
    save_path='best_model.pt'
)

# Save complete package
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'scaler': scaler,
    'train_losses': history['train_losses'],
    'val_losses': history['val_losses']
}
torch.save(checkpoint, 'temporal_complete.pt')
print("Model saved!")

# ==================== INFERENCE ====================

# Load model (in a new session)
checkpoint = torch.load('temporal_complete.pt')

# Recreate model
model = Temporal(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scaler
scaler = checkpoint['scaler']

# Make prediction
new_data = np.random.randn(96, 1)  # New historical data
new_data_normalized = scaler.transform(new_data)

# Convert to tensor
x = torch.FloatTensor(new_data_normalized).unsqueeze(0)

# Generate forecast
with torch.no_grad():
    forecast = model.forecast(x)

# Denormalize
forecast_np = forecast.cpu().numpy().reshape(-1, 1)
forecast_original = scaler.inverse_transform(forecast_np)

print(f"Forecast: {forecast_original.flatten()}")
```

## Common Issues

### Issue: "RuntimeError: Error(s) in loading state_dict"

**Cause**: Model architecture changed since saving

**Solution**:
```python
# Load with strict=False to ignore mismatched keys
model.load_state_dict(torch.load('model.pt'), strict=False)
```

### Issue: "Model outputs different results after loading"

**Cause**: Model not set to eval mode

**Solution**:
```python
model.eval()  # Always call this before inference
```

### Issue: "CUDA out of memory when loading"

**Cause**: Model saved on GPU, loading on different GPU

**Solution**:
```python
# Load to CPU first
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
# Then move to GPU if needed
model = model.to('cuda')
```

## Next Steps

- See `TRAINING_GUIDE.md` for training workflows
- See `examples/basic_usage.py` for complete training example
- Check PyTorch documentation on [serialization](https://pytorch.org/docs/stable/notes/serialization.html)
