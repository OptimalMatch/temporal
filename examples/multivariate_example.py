"""
Multivariate time series forecasting example.

This example demonstrates how to use Temporal for multivariate time series,
where multiple features are forecasted simultaneously.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer


def generate_multivariate_data(num_samples=1000, num_features=3):
    """
    Generate synthetic multivariate time series data.

    Simulates correlated features (e.g., temperature, humidity, pressure).
    """
    t = np.linspace(0, 100, num_samples)

    data = np.zeros((num_samples, num_features))

    # Feature 1: Temperature-like pattern
    data[:, 0] = (
        20 + 10 * np.sin(t * 0.5) +  # Seasonal variation
        2 * np.sin(t * 2) +  # Daily variation
        np.random.randn(num_samples) * 0.5
    )

    # Feature 2: Humidity-like pattern (negatively correlated with temperature)
    data[:, 1] = (
        60 - 5 * np.sin(t * 0.5) +  # Inverse seasonal
        3 * np.sin(t * 3) +  # Different frequency
        np.random.randn(num_samples) * 1.0
    )

    # Feature 3: Pressure-like pattern
    data[:, 2] = (
        1013 + 5 * np.cos(t * 0.5) +  # Slow variation
        np.random.randn(num_samples) * 0.3
    )

    return data


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    num_features = 3
    lookback = 96
    forecast_horizon = 24
    batch_size = 32
    num_epochs = 50

    # Generate data
    print("Generating multivariate time series data...")
    data = generate_multivariate_data(num_samples=1000, num_features=num_features)

    # Split data
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

    # Create model for multivariate data
    print("\nCreating Temporal model for multivariate forecasting...")
    model = Temporal(
        input_dim=num_features,  # Multiple features
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

    # Train
    print("\nTraining model...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=10,
        save_path="best_multivariate_model.pt"
    )

    # Visualize predictions for each feature
    print("\nGenerating forecasts...")

    sample_idx = 0
    src, decoder_input, target_output = val_dataset[sample_idx]

    with torch.no_grad():
        src_tensor = src.unsqueeze(0).to(trainer.device)
        pred = model.forecast(src_tensor)
        pred = pred.cpu().numpy().squeeze()

    feature_names = ['Temperature', 'Humidity', 'Pressure']

    fig, axes = plt.subplots(num_features, 1, figsize=(15, 12))

    for i in range(num_features):
        historical = src.numpy()[:, i]
        actual = target_output.numpy()[:, i]
        predicted = pred[:, i]

        axes[i].plot(range(len(historical)), historical,
                    label='Historical', color='blue')
        axes[i].plot(range(len(historical), len(historical) + len(actual)),
                    actual, label='Actual Future', color='green')
        axes[i].plot(range(len(historical), len(historical) + len(predicted)),
                    predicted, label='Predicted Future', color='red', linestyle='--')
        axes[i].axvline(x=len(historical), color='black', linestyle=':', alpha=0.5)
        axes[i].set_ylabel(feature_names[i])
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Steps')
    plt.suptitle('Multivariate Time Series Forecasting')
    plt.tight_layout()
    plt.savefig('multivariate_forecast.png')
    print("Multivariate forecast visualization saved to 'multivariate_forecast.png'")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
