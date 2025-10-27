# Data Sources Guide

This guide covers how to fetch and prepare real-world financial data for training Temporal models.

## Table of Contents
- [Installation](#installation)
- [Stock Data](#stock-data)
- [Cryptocurrency Data](#cryptocurrency-data)
- [Technical Indicators](#technical-indicators)
- [Data Preparation](#data-preparation)
- [Complete Examples](#complete-examples)

## Installation

Install Temporal with data fetching capabilities:

```bash
pip install temporal-forecasting[data]
```

This installs:
- `yfinance` - For fetching stock/crypto data from Yahoo Finance
- `pandas` - For data manipulation
- `scikit-learn` - For data preprocessing

## Stock Data

### Fetch Single Stock

```python
from temporal.data_sources import fetch_stock_data

# Fetch Apple stock data
df = fetch_stock_data('AAPL', period='2y', interval='1d')

print(df.head())
# Output:
#         Date        Open        High         Low       Close   Volume
# 0  2023-01-01   130.28    ...
```

### Parameters

- **ticker**: Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT', 'TSLA')
- **period**: Time period
  - Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
- **interval**: Data frequency
  - Options: '1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'
- **start_date**: Start date ('YYYY-MM-DD')
- **end_date**: End date ('YYYY-MM-DD')

### Examples

```python
# Get 5 years of daily data
df = fetch_stock_data('AAPL', period='5y', interval='1d')

# Get specific date range
df = fetch_stock_data('GOOGL', start_date='2023-01-01', end_date='2024-01-01')

# Get hourly data for last month
df = fetch_stock_data('TSLA', period='1mo', interval='1h')
```

### Fetch Multiple Stocks

```python
from temporal.data_sources import fetch_multiple_stocks

# Fetch multiple stocks at once
df = fetch_multiple_stocks(
    ['AAPL', 'GOOGL', 'MSFT'],
    period='1y',
    column='Close'  # Which column to extract
)

print(df.head())
# Output:
#         Date        AAPL      GOOGL      MSFT
# 0  2023-01-01   130.28    88.73     242.12
```

## Cryptocurrency Data

### Fetch Crypto Prices

```python
from temporal.data_sources import fetch_crypto_data

# Fetch Bitcoin
df = fetch_crypto_data('BTC-USD', period='2y', interval='1d')

# Fetch Ethereum
df = fetch_crypto_data('ETH-USD', period='1y', interval='1d')

# Fetch Dogecoin
df = fetch_crypto_data('DOGE-USD', period='6mo', interval='1d')
```

### Popular Crypto Symbols

- Bitcoin: `BTC-USD`
- Ethereum: `ETH-USD`
- Binance Coin: `BNB-USD`
- Cardano: `ADA-USD`
- Solana: `SOL-USD`
- Dogecoin: `DOGE-USD`
- Polygon: `MATIC-USD`
- Polkadot: `DOT-USD`

## Technical Indicators

Add technical indicators to enhance forecasting:

```python
from temporal.data_sources import fetch_stock_data, add_technical_indicators

# Fetch stock data
df = fetch_stock_data('AAPL', period='1y')

# Add technical indicators
df_with_indicators = add_technical_indicators(df)

print(df_with_indicators.columns)
# Output includes:
# - SMA_7, SMA_21, SMA_50 (Simple Moving Averages)
# - EMA_12, EMA_26 (Exponential Moving Averages)
# - MACD, MACD_Signal
# - RSI_14 (Relative Strength Index)
# - BB_Upper, BB_Middle, BB_Lower (Bollinger Bands)
# - Returns (Daily returns)
# - Volume_Change
```

### Using Indicators for Training

```python
from temporal.data_sources import prepare_for_temporal

# Prepare multivariate data with indicators
df_with_indicators = df_with_indicators.dropna()  # Remove NaN

data = prepare_for_temporal(
    df_with_indicators,
    feature_columns=['Close', 'Volume', 'SMA_7', 'RSI_14', 'MACD']
)

print(data.shape)  # (samples, 5 features)
```

## Data Preparation

### Convert to Training Format

```python
from temporal.data_sources import fetch_stock_data, prepare_for_temporal

# Fetch data
df = fetch_stock_data('AAPL', period='2y')

# Prepare for Temporal model (univariate)
data = prepare_for_temporal(df, feature_columns='Close')
print(data.shape)  # (num_days, 1)

# Prepare multivariate
data = prepare_for_temporal(
    df,
    feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']
)
print(data.shape)  # (num_days, 5)
```

### Normalize Data

```python
from temporal.data_sources import normalize_data

# Normalize with different methods
data_norm, scaler = normalize_data(data, method='standard')  # StandardScaler
# data_norm, scaler = normalize_data(data, method='minmax')   # MinMaxScaler
# data_norm, scaler = normalize_data(data, method='robust')   # RobustScaler

# Later: denormalize predictions
predictions_original = scaler.inverse_transform(predictions)
```

### Split Data

```python
from temporal.data_sources import split_train_val_test

# Split into train/val/test
train_data, val_data, test_data = split_train_val_test(
    data_normalized,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### Create Sequences

```python
from temporal.data_sources import create_sequences

# Create input-output sequences
X, y = create_sequences(
    data_normalized,
    lookback=60,         # Use 60 days
    forecast_horizon=5,  # Predict 5 days ahead
    stride=1            # Step by 1 day
)

print(X.shape)  # (num_sequences, 60, num_features)
print(y.shape)  # (num_sequences, 5, num_features)
```

## Complete Examples

### Example 1: Stock Forecasting Pipeline

```python
from temporal.data_sources import (
    fetch_stock_data,
    prepare_for_temporal,
    normalize_data,
    split_train_val_test
)
from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer
from torch.utils.data import DataLoader
import torch

# 1. Fetch data
df = fetch_stock_data('AAPL', period='2y', interval='1d')

# 2. Prepare
data = prepare_for_temporal(df, feature_columns='Close')

# 3. Normalize
data_norm, scaler = normalize_data(data, method='standard')

# 4. Split
train_data, val_data, test_data = split_train_val_test(data_norm)

# 5. Create datasets
train_dataset = TimeSeriesDataset(train_data, lookback=60, forecast_horizon=5)
val_dataset = TimeSeriesDataset(val_data, lookback=60, forecast_horizon=5)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. Create and train model
model = Temporal(input_dim=1, forecast_horizon=5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = TemporalTrainer(model=model, optimizer=optimizer)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10
)

# 7. Make predictions
with torch.no_grad():
    latest = torch.FloatTensor(data_norm[-60:]).unsqueeze(0)
    forecast = model.forecast(latest)

# 8. Denormalize
forecast_original = scaler.inverse_transform(
    forecast.cpu().numpy().reshape(-1, 1)
)

print("5-day forecast:", forecast_original.flatten())
```

### Example 2: Bitcoin Forecasting

```python
from temporal.data_sources import fetch_crypto_data, prepare_for_temporal

# Fetch Bitcoin data
df = fetch_crypto_data('BTC-USD', period='2y', interval='1d')

print(f"Latest Bitcoin price: ${df['Close'].iloc[-1]:,.2f}")

# Prepare for training
data = prepare_for_temporal(df, feature_columns='Close')

# Continue with training pipeline...
```

### Example 3: Multivariate with Indicators

```python
from temporal.data_sources import (
    fetch_stock_data,
    add_technical_indicators,
    prepare_for_temporal
)

# Fetch and add indicators
df = fetch_stock_data('AAPL', period='2y')
df = add_technical_indicators(df)
df = df.dropna()  # Remove NaN from indicators

# Use multiple features
data = prepare_for_temporal(
    df,
    feature_columns=[
        'Close',      # Price
        'Volume',     # Trading volume
        'SMA_7',      # 7-day moving average
        'RSI_14',     # Relative strength
        'MACD'        # MACD indicator
    ]
)

print(f"Data shape: {data.shape}")  # (samples, 5 features)

# Train multivariate model
model = Temporal(input_dim=5, forecast_horizon=5)
# ... training code ...
```

## Running Complete Examples

We provide ready-to-run examples:

### Stock Forecasting

```bash
cd examples
python stock_forecasting.py
```

This will:
- Fetch 2 years of Apple (AAPL) stock data
- Train a Temporal model
- Make 5-day forecasts
- Save visualizations

### Crypto Forecasting

```bash
cd examples
python crypto_forecasting.py
```

This will:
- Fetch Bitcoin (BTC-USD) data
- Train a model for 7-day forecasts
- Show predicted prices
- Save visualizations

### Customize for Your Needs

Edit the examples to:
- Change ticker symbols (line with `TICKER = 'AAPL'`)
- Adjust forecast horizon
- Add more features
- Experiment with model parameters

## Data Source Limitations

### Yahoo Finance Limits

- Historical data availability varies by ticker
- Some intervals have data limits:
  - 1-minute data: Last 7 days only
  - Hourly data: Last 730 days
  - Daily data: Full history available
- Data may have gaps (weekends, holidays, market closures)

### Handling Missing Data

```python
import pandas as pd

# Forward fill missing values
df = df.fillna(method='ffill')

# Or drop missing values
df = df.dropna()

# Or interpolate
df = df.interpolate(method='linear')
```

## Best Practices

### 1. Always Normalize

```python
# Financial data has different scales
# Always normalize before training
data_norm, scaler = normalize_data(data, method='standard')
```

### 2. Use Appropriate Lookback

```python
# For daily stock data:
lookback = 60   # 2-3 months of history
lookback = 252  # 1 year of trading days

# For hourly data:
lookback = 24 * 7  # 1 week

# For crypto (24/7 markets):
lookback = 24 * 30  # 1 month of hourly data
```

### 3. Handle Volatility

Crypto markets are more volatile than stocks:

```python
# Consider using robust scaling for crypto
data_norm, scaler = normalize_data(data, method='robust')

# Or use log returns
import numpy as np
returns = np.log(data[1:] / data[:-1])
```

### 4. Save Your Scaler

```python
import pickle

# Save scaler with model
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load for predictions
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

## Troubleshooting

### Issue: "yfinance not installed"

```bash
pip install yfinance
# Or
pip install temporal-forecasting[data]
```

### Issue: "No data found"

Check that:
- Ticker symbol is correct
- Date range is valid
- Market is open during specified period

```python
# Check available data
import yfinance as yf
ticker = yf.Ticker('AAPL')
print(ticker.history(period='max').index)
```

### Issue: "Too many NaN values"

```python
# Some indicators need minimum data
# Use longer period or drop early rows
df = df.dropna()

# Or check minimum required length
min_length = 50  # For 50-day SMA
if len(df) < min_length:
    print(f"Need at least {min_length} days of data")
```

## Resources

- [Yahoo Finance](https://finance.yahoo.com) - Search for ticker symbols
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Technical Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)

## Next Steps

- See `examples/stock_forecasting.py` for complete stock example
- See `examples/crypto_forecasting.py` for complete crypto example
- Check `TRAINING_GUIDE.md` for general training tips
- Read `MODEL_PERSISTENCE.md` for saving models
