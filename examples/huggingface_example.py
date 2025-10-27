"""
HuggingFace Integration Example

Demonstrates how to:
1. Create a HuggingFace-compatible Temporal model
2. Train it
3. Save it in HF format
4. Load it back
5. (Optional) Upload to HuggingFace Hub
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import json
import pickle
import sys
sys.path.append('..')

# Check if HuggingFace libraries are available
try:
    from temporal.hf_interface import TemporalForForecasting, TemporalConfig, create_hf_model
    from temporal.trainer import TimeSeriesDataset, TemporalTrainer
    HF_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    print("\nHuggingFace integration requires additional packages.")
    print("Install with: pip install transformers huggingface-hub")
    HF_AVAILABLE = False
    sys.exit(1)


def train_hf_model():
    """
    Train a HuggingFace-compatible Temporal model.
    """
    print("=" * 60)
    print("TRAINING HUGGINGFACE-COMPATIBLE MODEL")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    t = np.linspace(0, 100, 1000)
    data = (
        np.sin(t * 0.5) +
        0.5 * np.sin(t * 2) +
        0.3 * np.sin(t * 5) +
        0.1 * np.random.randn(1000)
    ).reshape(-1, 1)

    # Normalize
    print("2. Normalizing data...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Split
    train_size = int(0.8 * len(data_normalized))
    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:]

    # Create datasets
    print("3. Creating datasets...")
    train_dataset = TimeSeriesDataset(train_data, lookback=96, forecast_horizon=24)
    val_dataset = TimeSeriesDataset(val_data, lookback=96, forecast_horizon=24)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create HF-compatible model
    print("4. Creating HuggingFace-compatible model...")
    config = TemporalConfig(
        input_dim=1,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=512,
        forecast_horizon=24,
        dropout=0.1
    )

    model = TemporalForForecasting(config)
    print(f"   Model type: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n5. Training model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Note: Trainer expects the base Temporal model, not the HF wrapper
    trainer = TemporalTrainer(
        model=model.temporal,  # Use underlying Temporal model
        optimizer=optimizer,
        criterion=torch.nn.MSELoss()
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,  # Reduced for demo
        early_stopping_patience=5
    )

    # Save in HuggingFace format
    print("\n6. Saving in HuggingFace format...")
    save_dir = "./temporal-hf-model"

    # Save model (HF format)
    model.save_pretrained(save_dir)
    print(f"   ✓ Model saved to {save_dir}")

    # Save scaler
    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved")

    # Save training info
    training_info = {
        'epochs': len(history['train_losses']),
        'best_val_loss': min(history['val_losses']),
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'lookback': 96,
        'forecast_horizon': 24
    }

    with open(f"{save_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"   ✓ Training info saved")

    print("\n✓ Training completed!")
    return save_dir, scaler


def load_and_test_hf_model(save_dir, original_scaler):
    """
    Load and test the HuggingFace model.
    """
    print("\n" + "=" * 60)
    print("LOADING HUGGINGFACE MODEL")
    print("=" * 60)

    # Load model
    print("\n1. Loading model from HuggingFace format...")
    model = TemporalForForecasting.from_pretrained(save_dir)
    model.eval()
    print(f"   ✓ Model loaded")
    print(f"   Model type: {type(model).__name__}")

    # Load scaler
    print("\n2. Loading scaler...")
    with open(f"{save_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print(f"   ✓ Scaler loaded")

    # Load training info
    print("\n3. Loading training info...")
    with open(f"{save_dir}/training_info.json", "r") as f:
        training_info = json.load(f)

    print(f"   Training epochs: {training_info['epochs']}")
    print(f"   Best val loss: {training_info['best_val_loss']:.6f}")
    print(f"   Lookback: {training_info['lookback']}")
    print(f"   Forecast horizon: {training_info['forecast_horizon']}")

    # Test prediction
    print("\n4. Testing prediction...")
    test_data = np.sin(np.linspace(0, 10, 96)).reshape(-1, 1)
    test_data_normalized = scaler.transform(test_data)

    x = torch.FloatTensor(test_data_normalized).unsqueeze(0)

    with torch.no_grad():
        forecast = model.forecast(x)

    # Denormalize
    forecast_np = forecast.cpu().numpy().reshape(-1, 1)
    forecast_original = scaler.inverse_transform(forecast_np)

    print(f"   Input shape: {x.shape}")
    print(f"   Forecast shape: {forecast.shape}")
    print(f"   Sample forecast values: {forecast_original.flatten()[:5]}")

    print("\n✓ Loading and testing completed!")


def demonstrate_hub_upload(save_dir):
    """
    Demonstrate how to upload to HuggingFace Hub (optional).
    """
    print("\n" + "=" * 60)
    print("HUGGINGFACE HUB UPLOAD (Optional)")
    print("=" * 60)

    print("\nTo upload your model to HuggingFace Hub:")
    print("\n1. Create an account at https://huggingface.co/join")
    print("2. Get your access token from https://huggingface.co/settings/tokens")
    print("3. Login:")
    print("   huggingface-cli login")
    print("   # Or in Python:")
    print("   from huggingface_hub import login")
    print("   login(token='your_token')")
    print("\n4. Upload:")
    print("   from temporal.hf_interface import TemporalForForecasting")
    print(f"   model = TemporalForForecasting.from_pretrained('{save_dir}')")
    print("   model.push_to_hub('your-username/temporal-forecasting')")
    print("\n5. Share with others:")
    print("   model = TemporalForForecasting.from_pretrained('your-username/temporal-forecasting')")

    print("\n" + "=" * 60)


def create_model_card(save_dir):
    """
    Create a model card (README.md) for the model.
    """
    print("\n" + "=" * 60)
    print("CREATING MODEL CARD")
    print("=" * 60)

    # Load training info
    with open(f"{save_dir}/training_info.json", "r") as f:
        training_info = json.load(f)

    model_card = f"""---
license: mit
tags:
- time-series
- forecasting
- transformer
- temporal
library_name: temporal-forecasting
---

# Temporal Time Series Forecasting Model

## Model Description

This is a Temporal transformer model for time series forecasting, trained on synthetic sine wave data.

## Model Details

- **Model Type**: Temporal Transformer
- **Architecture**: Encoder-Decoder with Multi-Head Attention
- **Input Dimension**: 1 (univariate)
- **Lookback Window**: {training_info['lookback']} time steps
- **Forecast Horizon**: {training_info['forecast_horizon']} time steps
- **Model Size**: 128 dimensions, 2 encoder layers, 2 decoder layers

## Training

- **Training Epochs**: {training_info['epochs']}
- **Best Validation Loss**: {training_info['best_val_loss']:.6f}
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3

## How to Use

```python
from temporal.hf_interface import TemporalForForecasting
import torch
import pickle

# Load model
model = TemporalForForecasting.from_pretrained("./temporal-hf-model")

# Load scaler (if saved)
with open("./temporal-hf-model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare your data (96 historical time steps)
import numpy as np
data = np.random.randn(96, 1)
data_normalized = scaler.transform(data)

x = torch.FloatTensor(data_normalized).unsqueeze(0)

# Generate forecast (24 future time steps)
forecast = model.forecast(x)

# Denormalize
forecast_np = forecast.cpu().numpy().reshape(-1, 1)
forecast_original = scaler.inverse_transform(forecast_np)

print(forecast_original)
```

## Limitations

- Trained on synthetic data only
- Best suited for similar sine wave patterns
- May require fine-tuning for real-world data

## Installation

```bash
pip install temporal-forecasting transformers huggingface-hub
```

## Citation

```bibtex
@software{{temporal_model,
  title = {{Temporal Time Series Forecasting}},
  year = {{2024}},
  url = {{https://github.com/OptimalMatch/temporal}}
}}
```

## License

MIT License
"""

    # Save model card
    readme_path = f"{save_dir}/README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)

    print(f"\n✓ Model card created at {readme_path}")
    print("\nModel card preview:")
    print("-" * 60)
    print(model_card[:500] + "...")
    print("-" * 60)


def main():
    """
    Main function demonstrating HuggingFace integration.
    """
    print("\n" + "=" * 60)
    print("HUGGINGFACE INTEGRATION EXAMPLE")
    print("=" * 60)

    if not HF_AVAILABLE:
        print("\nError: HuggingFace libraries not available")
        return

    # Step 1: Train and save
    save_dir, scaler = train_hf_model()

    # Step 2: Create model card
    create_model_card(save_dir)

    # Step 3: Load and test
    load_and_test_hf_model(save_dir, scaler)

    # Step 4: Show upload instructions
    demonstrate_hub_upload(save_dir)

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nModel saved in HuggingFace format at: {save_dir}")
    print("\nFiles created:")
    print(f"  - {save_dir}/config.json")
    print(f"  - {save_dir}/pytorch_model.bin")
    print(f"  - {save_dir}/scaler.pkl")
    print(f"  - {save_dir}/training_info.json")
    print(f"  - {save_dir}/README.md")
    print("\nYou can now:")
    print("  1. Share the model directory with others")
    print("  2. Upload to HuggingFace Hub (see instructions above)")
    print("  3. Load the model anywhere with TemporalForForecasting.from_pretrained()")


if __name__ == "__main__":
    main()
