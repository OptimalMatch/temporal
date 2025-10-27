"""
Model Persistence Example

Demonstrates how to:
1. Train a model
2. Save it with all necessary components
3. Load it in a new session
4. Use it for inference
"""

import torch
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import sys
sys.path.append('..')

from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer


class ModelManager:
    """
    Manages saving and loading of Temporal models.
    """

    @staticmethod
    def save_model(model, config, scaler, save_path, training_info=None):
        """
        Save model with all necessary components.

        Args:
            model: Trained Temporal model
            config: Model configuration dict
            scaler: Fitted data scaler
            save_path: Path to save model
            training_info: Optional dict with training history
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'scaler': scaler,
            'pytorch_version': torch.__version__
        }

        if training_info:
            checkpoint['training_info'] = training_info

        torch.save(checkpoint, save_path)
        print(f"\n✓ Model saved to {save_path}")

        # Also save config as JSON for easy reference
        config_path = save_path.replace('.pt', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Config saved to {config_path}")

    @staticmethod
    def load_model(load_path, device='cpu'):
        """
        Load model from checkpoint.

        Args:
            load_path: Path to saved model
            device: Device to load model on

        Returns:
            model, config, scaler
        """
        checkpoint = torch.load(load_path, map_location=device)

        # Create model
        model = Temporal(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)

        print(f"\n✓ Model loaded from {load_path}")
        print(f"  PyTorch version: {checkpoint.get('pytorch_version', 'unknown')}")

        if 'training_info' in checkpoint:
            info = checkpoint['training_info']
            print(f"  Training epochs: {info.get('epochs', 'unknown')}")
            print(f"  Best val loss: {info.get('best_val_loss', 'unknown'):.6f}")

        return model, checkpoint['config'], checkpoint['scaler']


def train_and_save():
    """
    Train a model and save it.
    """
    print("=" * 60)
    print("TRAINING AND SAVING MODEL")
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

    # Normalize data
    print("2. Normalizing data...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Split data
    train_size = int(0.8 * len(data_normalized))
    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:]

    # Create datasets
    print("3. Creating datasets...")
    train_dataset = TimeSeriesDataset(
        train_data,
        lookback=96,
        forecast_horizon=24,
        stride=1
    )
    val_dataset = TimeSeriesDataset(
        val_data,
        lookback=96,
        forecast_horizon=24,
        stride=1
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    # Model configuration
    config = {
        'input_dim': 1,
        'd_model': 128,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'num_heads': 4,
        'd_ff': 512,
        'forecast_horizon': 24,
        'dropout': 0.1
    }

    # Create model
    print("\n4. Creating model...")
    model = Temporal(**config)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_clip=1.0
    )

    # Train model
    print("\n5. Training model...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        early_stopping_patience=5,
        save_path="best_model_temp.pt"
    )

    # Prepare training info
    training_info = {
        'epochs': len(history['train_losses']),
        'best_val_loss': min(history['val_losses']),
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses']
    }

    # Save complete model
    print("\n6. Saving model...")
    ModelManager.save_model(
        model=model,
        config=config,
        scaler=scaler,
        save_path='saved_temporal_model.pt',
        training_info=training_info
    )

    print("\n✓ Training and saving completed!")
    return config


def load_and_predict(config):
    """
    Load a saved model and make predictions.
    """
    print("\n" + "=" * 60)
    print("LOADING AND USING SAVED MODEL")
    print("=" * 60)

    # Load model
    print("\n1. Loading saved model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, loaded_config, scaler = ModelManager.load_model(
        'saved_temporal_model.pt',
        device=device
    )

    # Verify config matches
    print("\n2. Verifying model configuration...")
    for key in config:
        if config[key] == loaded_config[key]:
            print(f"   ✓ {key}: {config[key]}")
        else:
            print(f"   ✗ {key}: {config[key]} != {loaded_config[key]}")

    # Generate test data
    print("\n3. Generating test data...")
    test_data = np.sin(np.linspace(0, 10, 96)).reshape(-1, 1)
    print(f"   Test data shape: {test_data.shape}")

    # Normalize test data
    test_data_normalized = scaler.transform(test_data)

    # Convert to tensor
    x = torch.FloatTensor(test_data_normalized).unsqueeze(0).to(device)
    print(f"   Input tensor shape: {x.shape}")

    # Make prediction
    print("\n4. Making prediction...")
    with torch.no_grad():
        forecast = model.forecast(x)

    print(f"   Forecast tensor shape: {forecast.shape}")

    # Denormalize forecast
    forecast_np = forecast.cpu().numpy().reshape(-1, 1)
    forecast_original = scaler.inverse_transform(forecast_np)

    print("\n5. Forecast results:")
    print(f"   Normalized forecast: {forecast_np.flatten()[:5]}... (first 5 values)")
    print(f"   Original scale forecast: {forecast_original.flatten()[:5]}... (first 5 values)")

    print("\n✓ Inference completed successfully!")

    return forecast_original


def production_inference_example():
    """
    Example of using model in production setting.
    """
    print("\n" + "=" * 60)
    print("PRODUCTION INFERENCE EXAMPLE")
    print("=" * 60)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, scaler = ModelManager.load_model(
        'saved_temporal_model.pt',
        device=device
    )

    # Simulate incoming data stream
    print("\nSimulating real-time predictions...")

    for i in range(3):
        # New data arrives
        new_data = np.random.randn(96, 1)

        # Preprocess
        new_data_normalized = scaler.transform(new_data)

        # Convert to tensor
        x = torch.FloatTensor(new_data_normalized).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            forecast = model.forecast(x)

        # Postprocess
        forecast_np = forecast.cpu().numpy().reshape(-1, 1)
        forecast_original = scaler.inverse_transform(forecast_np)

        print(f"\nPrediction {i+1}:")
        print(f"  Input shape: {new_data.shape}")
        print(f"  Forecast shape: {forecast_original.shape}")
        print(f"  Forecast values: {forecast_original.flatten()[:3]}...")


def main():
    """
    Main function demonstrating complete workflow.
    """
    print("\n" + "=" * 60)
    print("MODEL PERSISTENCE EXAMPLE")
    print("=" * 60)

    # Step 1: Train and save model
    config = train_and_save()

    # Step 2: Load and use model
    load_and_predict(config)

    # Step 3: Production inference example
    production_inference_example()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nSaved files:")
    print("  - saved_temporal_model.pt (complete model package)")
    print("  - saved_temporal_model_config.json (human-readable config)")
    print("\nTo use the model in another script:")
    print("  from examples.model_persistence_example import ModelManager")
    print("  model, config, scaler = ModelManager.load_model('saved_temporal_model.pt')")


if __name__ == "__main__":
    main()
