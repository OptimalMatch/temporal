"""
Basic usage example for Temporal model.

This example demonstrates how to:
1. Create a Temporal model
2. Generate synthetic time series data
3. Train the model
4. Make forecasts
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer


def generate_synthetic_data(num_samples=1000, seq_len=500):
    """
    Generate synthetic time series data for demonstration.

    Creates a combination of sine waves with noise.
    """
    t = np.linspace(0, 100, num_samples)

    # Combine multiple seasonal patterns
    data = (
        np.sin(t * 0.5) +  # Long-term trend
        0.5 * np.sin(t * 2) +  # Medium-term pattern
        0.3 * np.sin(t * 5) +  # Short-term pattern
        0.1 * np.random.randn(num_samples)  # Noise
    )

    # Reshape to (num_samples, 1) for univariate time series
    data = data.reshape(-1, 1)

    return data


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    lookback = 96  # Use 96 time steps to predict future
    forecast_horizon = 24  # Predict 24 steps ahead
    batch_size = 32
    num_epochs = 50

    # Generate synthetic data
    print("Generating synthetic time series data...")
    data = generate_synthetic_data(num_samples=1000)

    # Split into train and validation
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        stride=1
    )

    val_dataset = TimeSeriesDataset(
        val_data,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        stride=1
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\nCreating Temporal model...")
    model = Temporal(
        input_dim=1,
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        d_ff=1024,
        forecast_horizon=forecast_horizon,
        dropout=0.1
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Create trainer
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_clip=1.0
    )

    # Train model
    print("\nTraining model...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=10,
        save_path="best_model.pt"
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print("\nTraining history saved to 'training_history.png'")

    # Make predictions
    print("\nGenerating forecasts...")
    predictions = trainer.predict(val_loader)

    # Visualize some predictions
    plt.figure(figsize=(15, 5))

    # Get a sample from validation data
    sample_idx = 0
    src, decoder_input, target_output = val_dataset[sample_idx]

    # Make prediction for this sample
    with torch.no_grad():
        src_tensor = src.unsqueeze(0).to(trainer.device)
        pred = model.forecast(src_tensor)
        pred = pred.cpu().numpy().squeeze()

    # Plot
    historical = src.numpy().squeeze()
    actual = target_output.numpy().squeeze()

    plt.plot(range(len(historical)), historical, label='Historical', color='blue')
    plt.plot(range(len(historical), len(historical) + len(actual)),
             actual, label='Actual Future', color='green')
    plt.plot(range(len(historical), len(historical) + len(pred)),
             pred, label='Predicted Future', color='red', linestyle='--')
    plt.axvline(x=len(historical), color='black', linestyle=':', alpha=0.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Time Series Forecast')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_example.png')
    print("Forecast visualization saved to 'forecast_example.png'")

    # Calculate metrics
    mse = np.mean((predictions.squeeze() - val_data[lookback:lookback+len(predictions)].squeeze()) ** 2)
    mae = np.mean(np.abs(predictions.squeeze() - val_data[lookback:lookback+len(predictions)].squeeze()))

    print(f"\nValidation Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")


if __name__ == "__main__":
    main()
