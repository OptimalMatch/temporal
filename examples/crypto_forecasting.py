"""
Cryptocurrency Price Forecasting Example

Demonstrates how to:
1. Fetch Bitcoin and other cryptocurrency data
2. Train the Temporal model for crypto price prediction
3. Compare multiple cryptocurrencies
4. Handle high volatility in crypto markets
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer
from temporal.data_sources import (
    fetch_crypto_data,
    prepare_for_temporal,
    split_train_val_test
)


def train_crypto_model(
    symbol='BTC-USD',
    period='2y',
    lookback=60,
    forecast_horizon=7,
    epochs=30
):
    """
    Train a model for cryptocurrency price forecasting.

    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
        period: Historical period to fetch
        lookback: Days of history to use
        forecast_horizon: Days to forecast ahead
        epochs: Training epochs

    Returns:
        Trained model, scaler, history
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL FOR {symbol}")
    print(f"{'='*70}")

    # Fetch data
    print(f"\n1. Fetching {symbol} data...")
    df = fetch_crypto_data(symbol, period=period, interval='1d')
    print(f"   ✓ Fetched {len(df)} days")
    print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"   Latest price: ${df['Close'].iloc[-1]:,.2f}")

    # Prepare data
    print(f"\n2. Preparing data...")
    data = prepare_for_temporal(df, feature_columns='Close')
    print(f"   Shape: {data.shape}")

    # Normalize
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Split
    train_data, val_data, test_data = split_train_val_test(data_normalized)
    print(f"   Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, lookback, forecast_horizon)
    val_dataset = TimeSeriesDataset(val_data, lookback, forecast_horizon)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    print(f"\n3. Creating model...")
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

    # Train
    print(f"\n4. Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_clip=1.0
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        early_stopping_patience=10,
        save_path=f"{symbol.replace('-', '_')}_model.pt"
    )

    print(f"\n   ✓ Training completed!")
    print(f"   Best val loss: {min(history['val_losses']):.6f}")

    return model, scaler, history, trainer


def compare_multiple_cryptos():
    """
    Compare forecasting performance across multiple cryptocurrencies.
    """
    print("\n" + "=" * 70)
    print("MULTI-CRYPTOCURRENCY COMPARISON")
    print("=" * 70)

    cryptos = [
        ('BTC-USD', 'Bitcoin'),
        ('ETH-USD', 'Ethereum'),
        ('BNB-USD', 'Binance Coin')
    ]

    results = {}

    for symbol, name in cryptos:
        print(f"\n\nProcessing {name} ({symbol})...")
        try:
            model, scaler, history, trainer = train_crypto_model(
                symbol=symbol,
                period='1y',
                lookback=30,
                forecast_horizon=7,
                epochs=20
            )
            results[name] = {
                'symbol': symbol,
                'model': model,
                'scaler': scaler,
                'history': history,
                'trainer': trainer
            }
        except Exception as e:
            print(f"   ✗ Error processing {name}: {e}")
            continue

    # Visualize comparison
    if results:
        print(f"\n\nCreating comparison visualization...")
        fig, axes = plt.subplots(len(results), 2, figsize=(15, 5 * len(results)))

        if len(results) == 1:
            axes = axes.reshape(1, -1)

        for idx, (name, data) in enumerate(results.items()):
            # Plot training history
            axes[idx, 0].plot(data['history']['train_losses'], label='Train', alpha=0.7)
            axes[idx, 0].plot(data['history']['val_losses'], label='Val', alpha=0.7)
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Loss')
            axes[idx, 0].set_title(f'{name} Training History')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            # Plot sample forecast
            # Fetch recent data for prediction
            df = fetch_crypto_data(data['symbol'], period='6mo')
            recent_data = prepare_for_temporal(df, 'Close')
            recent_normalized = data['scaler'].transform(recent_data)

            latest = recent_normalized[-30:]
            latest_tensor = torch.FloatTensor(latest).unsqueeze(0).to(data['trainer'].device)

            with torch.no_grad():
                forecast = data['model'].forecast(latest_tensor)

            forecast_original = data['scaler'].inverse_transform(
                forecast.cpu().numpy().reshape(-1, 1)
            ).flatten()

            # Plot
            hist_days = 30
            hist_prices = recent_data[-hist_days:, 0]
            axes[idx, 1].plot(np.arange(-hist_days, 0), hist_prices,
                            'b-', label='Historical', linewidth=2)
            axes[idx, 1].plot(np.arange(0, 7), forecast_original,
                            'r--', label='Forecast', linewidth=2, marker='o')
            axes[idx, 1].axvline(x=0, color='black', linestyle=':', alpha=0.5)
            axes[idx, 1].set_xlabel('Days')
            axes[idx, 1].set_ylabel('Price ($)')
            axes[idx, 1].set_title(f'{name} 7-Day Forecast')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('crypto_comparison.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved to 'crypto_comparison.png'")


def main():
    """
    Main function demonstrating crypto forecasting.
    """
    print("=" * 70)
    print("CRYPTOCURRENCY PRICE FORECASTING WITH TEMPORAL")
    print("=" * 70)

    # Example 1: Bitcoin forecasting
    print("\n\nEXAMPLE 1: BITCOIN PRICE FORECASTING")
    model, scaler, history, trainer = train_crypto_model(
        symbol='BTC-USD',
        period='2y',
        lookback=60,
        forecast_horizon=7,
        epochs=30
    )

    # Make future prediction
    print(f"\n5. Making 7-day Bitcoin price prediction...")
    df_latest = fetch_crypto_data('BTC-USD', period='3mo')
    data_latest = prepare_for_temporal(df_latest, 'Close')
    data_latest_norm = scaler.transform(data_latest)

    latest = data_latest_norm[-60:]
    latest_tensor = torch.FloatTensor(latest).unsqueeze(0).to(trainer.device)

    with torch.no_grad():
        forecast = model.forecast(latest_tensor)

    forecast_original = scaler.inverse_transform(
        forecast.cpu().numpy().reshape(-1, 1)
    ).flatten()

    current_price = data_latest[-1, 0]
    print(f"\n   Current BTC Price: ${current_price:,.2f}")
    print(f"\n   7-Day Forecast:")
    for i, price in enumerate(forecast_original, 1):
        change = ((price - current_price) / current_price) * 100
        direction = "📈" if change > 0 else "📉"
        print(f"   Day {i}: ${price:,.2f} ({direction} {change:+.2f}%)")

    # Visualize
    plt.figure(figsize=(15, 5))

    # Training history
    plt.subplot(1, 3, 1)
    plt.plot(history['train_losses'], label='Train Loss', alpha=0.7)
    plt.plot(history['val_losses'], label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bitcoin Model Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Price history
    plt.subplot(1, 3, 2)
    days_to_show = 90
    plt.plot(data_latest[-days_to_show:, 0])
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title('Bitcoin Price (Last 90 Days)')
    plt.grid(True, alpha=0.3)

    # Forecast
    plt.subplot(1, 3, 3)
    hist_days = 30
    hist_prices = data_latest[-hist_days:, 0]
    plt.plot(np.arange(-hist_days, 0), hist_prices, 'b-', label='Historical', linewidth=2)
    plt.plot(np.arange(0, 7), forecast_original, 'r--', label='Forecast',
             linewidth=2, marker='o', markersize=6)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title('Bitcoin 7-Day Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bitcoin_forecast.png', dpi=150, bbox_inches='tight')
    print(f"\n   ✓ Saved visualization to 'bitcoin_forecast.png'")

    # Example 2: Compare multiple cryptos (optional)
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: MULTI-CRYPTO COMPARISON (Optional)")
    print("=" * 70)
    print("\nUncomment the line below to compare BTC, ETH, and BNB:")
    print("# compare_multiple_cryptos()")

    # Uncomment to run:
    # compare_multiple_cryptos()

    print("\n" + "=" * 70)
    print("CRYPTO FORECASTING COMPLETED!")
    print("=" * 70)
    print("\nFiles created:")
    print("  - BTC_USD_model.pt (saved model)")
    print("  - bitcoin_forecast.png (visualization)")
    print("\nTry experimenting with:")
    print("  - Different cryptocurrencies (ETH-USD, BNB-USD, DOGE-USD, etc.)")
    print("  - Different forecast horizons")
    print("  - Adding volume or other features")


if __name__ == "__main__":
    main()
