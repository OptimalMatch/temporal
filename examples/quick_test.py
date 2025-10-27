"""
Quick test to verify Temporal installation and basic functionality.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from temporal import Temporal, TemporalTrainer, TimeSeriesDataset


def generate_synthetic_data(num_samples=500):
    """Generate simple synthetic time series."""
    t = np.linspace(0, 50, num_samples)
    data = (
        np.sin(t * 0.5) +
        0.5 * np.sin(t * 2) +
        0.1 * np.random.randn(num_samples)
    )
    return data.reshape(-1, 1)


def main():
    print("=" * 60)
    print("Temporal Model - Quick Test")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    lookback = 48
    forecast_horizon = 12
    batch_size = 16
    num_epochs = 5  # Quick test

    # Generate data
    print("\n1. Generating synthetic time series data...")
    data = generate_synthetic_data(num_samples=500)
    print(f"   Data shape: {data.shape}")

    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create datasets
    print("\n2. Creating datasets...")
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    # Create model
    print("\n3. Creating Temporal model...")
    model = Temporal(
        input_dim=1,
        d_model=128,  # Smaller for faster testing
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=512,
        forecast_horizon=forecast_horizon,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device=device,
        grad_clip=1.0
    )

    # Train
    print(f"\n4. Training model for {num_epochs} epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=10,
        save_path=None
    )

    # Test predictions
    print("\n5. Testing predictions...")
    predictions = trainer.predict(val_loader)
    print(f"   Predictions shape: {predictions.shape}")

    # Calculate metrics
    from temporal import calculate_metrics

    # Flatten predictions and get corresponding actual values
    # predictions shape: (num_samples, forecast_horizon, features)
    # We need to flatten and compare with actual future values
    pred_flat = predictions.reshape(-1, 1).squeeze()

    # Get actual values for all forecast horizons
    actual_list = []
    for i in range(len(predictions)):
        start_idx = lookback + i
        end_idx = start_idx + forecast_horizon
        actual_list.append(val_data[start_idx:end_idx].squeeze())
    actual_flat = np.concatenate(actual_list)

    metrics = calculate_metrics(pred_flat, actual_flat)

    print("\n6. Results:")
    print(f"   Final Training Loss: {history['train_losses'][-1]:.6f}")
    print(f"   Final Validation Loss: {history['val_losses'][-1]:.6f}")
    print(f"   Test MSE: {metrics['MSE']:.6f}")
    print(f"   Test MAE: {metrics['MAE']:.6f}")
    print(f"   Test R²: {metrics['R2']:.6f}")

    # Test single forecast
    print("\n7. Testing single forecast...")
    with torch.no_grad():
        sample = torch.FloatTensor(train_data[-lookback:]).unsqueeze(0).to(device)
        forecast = model.forecast(sample)
        print(f"   Forecast shape: {forecast.shape}")
        print(f"   Forecast values (first 5): {forecast[0, :5, 0].cpu().numpy()}")

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
