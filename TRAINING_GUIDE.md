# Training Guide for Temporal

This guide covers how to prepare datasets and train the Temporal model for various time series forecasting scenarios.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Training Workflow](#training-workflow)
- [Data Sources](#data-sources)
- [Best Practices](#best-practices)

## Dataset Preparation

### Understanding the Data Format

The `TimeSeriesDataset` class expects data in the shape: `(num_samples, num_features)`

- **Univariate**: `(num_samples, 1)` - single feature (e.g., temperature)
- **Multivariate**: `(num_samples, n)` - multiple features (e.g., temperature, humidity, pressure)

### Key Parameters

```python
TimeSeriesDataset(
    data,                    # Your time series data
    lookback=96,            # Number of historical steps to use
    forecast_horizon=24,    # Number of future steps to predict
    stride=1                # Step size for sliding window
)
```

**Example**: With `lookback=96`, `forecast_horizon=24`, and `stride=1`:
- Input: 96 time steps of historical data
- Output: Predict next 24 time steps
- Each sample shifts forward by 1 time step

## Training Workflow

### Complete Training Example

```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer

# 1. Prepare your data (example with random data)
data = np.random.randn(1000, 1)  # 1000 samples, univariate

# 2. Normalize data (recommended)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 3. Split into train/validation
train_size = int(0.8 * len(data_normalized))
train_data = data_normalized[:train_size]
val_data = data_normalized[train_size:]

# 4. Create datasets
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

# 5. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. Create model
model = Temporal(
    input_dim=1,              # Number of features
    d_model=256,              # Model dimension
    num_encoder_layers=4,     # Encoder depth
    num_decoder_layers=4,     # Decoder depth
    num_heads=8,              # Attention heads
    d_ff=1024,                # Feed-forward dimension
    forecast_horizon=24,      # Prediction horizon
    dropout=0.1
)

# 7. Create optimizer and trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

trainer = TemporalTrainer(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.MSELoss(),
    device="cuda" if torch.cuda.is_available() else "cpu",
    grad_clip=1.0
)

# 8. Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    early_stopping_patience=10,
    save_path="best_model.pt"
)

# 9. Make predictions
predictions = trainer.predict(val_loader)

# 10. Denormalize predictions
predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1))
```

## Data Sources

### 1. CSV Files

```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('timeseries.csv')

# Extract time series column(s)
# Univariate
data = df['temperature'].values.reshape(-1, 1)

# Multivariate
data = df[['temperature', 'humidity', 'pressure']].values

# Convert to numpy array with proper shape
data = np.array(data, dtype=np.float32)
```

**Example CSV structure**:
```csv
timestamp,temperature,humidity,pressure
2024-01-01 00:00:00,20.5,65,1013
2024-01-01 01:00:00,20.3,66,1013
...
```

### 2. NumPy Arrays

```python
# Load from .npy file
data = np.load('timeseries.npy')

# Ensure correct shape
if data.ndim == 1:
    data = data.reshape(-1, 1)  # Make it (num_samples, 1)
```

### 3. Pandas DataFrames

```python
import pandas as pd

# From DataFrame
df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, ...]
})

# Convert to numpy
data = df['value'].values.reshape(-1, 1)

# For datetime-indexed data
df_indexed = df.set_index('timestamp')
data = df_indexed.values
```

### 4. Real-Time Streaming Data

```python
# Collect data into a buffer
buffer = []

# As data arrives
for new_value in stream:
    buffer.append(new_value)

    # Once you have enough data
    if len(buffer) >= 1000:
        data = np.array(buffer).reshape(-1, 1)
        # Train or predict
        break
```

### 5. Database Queries

```python
import sqlalchemy as db
import pandas as pd

# Connect to database
engine = db.create_engine('postgresql://user:password@localhost/db')

# Query data
query = "SELECT timestamp, value FROM timeseries ORDER BY timestamp"
df = pd.read_sql(query, engine)

# Convert to numpy
data = df['value'].values.reshape(-1, 1)
```

### 6. API Data

```python
import requests
import json
import numpy as np

# Fetch from API
response = requests.get('https://api.example.com/timeseries')
json_data = response.json()

# Extract values
values = [item['value'] for item in json_data['data']]
data = np.array(values).reshape(-1, 1)
```

## Data Preprocessing

### Normalization (Highly Recommended)

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# MinMaxScaler (scale to [0, 1])
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Don't forget to save the scaler for inverse transform
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### Handling Missing Values

```python
import pandas as pd

# Forward fill
df = pd.DataFrame(data)
df_filled = df.fillna(method='ffill')

# Backward fill
df_filled = df.fillna(method='bfill')

# Interpolation
df_filled = df.interpolate(method='linear')

data = df_filled.values
```

### Handling Irregular Time Series

```python
import pandas as pd

# Resample to regular intervals
df = pd.DataFrame(data, index=timestamps)
df_regular = df.resample('1H').mean()  # Hourly resampling
df_regular = df_regular.fillna(method='ffill')

data = df_regular.values
```

## Best Practices

### 1. Data Splitting

```python
# Temporal split (no shuffling for time series!)
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]
```

### 2. Choosing Hyperparameters

**Lookback Window**:
- Short-term patterns: `lookback=24-96`
- Long-term patterns: `lookback=96-365`
- Rule of thumb: 3-5x the forecast horizon

**Forecast Horizon**:
- Depends on your use case
- Common: 1, 6, 12, 24, 48 steps

**Model Size**:
- Small data (<10k samples): `d_model=128`, `num_layers=2`
- Medium data (10k-100k): `d_model=256`, `num_layers=4`
- Large data (>100k): `d_model=512`, `num_layers=6`

### 3. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# In training loop
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)
    scheduler.step(val_loss)  # Reduce LR if validation loss plateaus
```

### 4. Batch Size Selection

```python
# Start with these and adjust based on GPU memory
batch_sizes = {
    'small_model': 64,
    'medium_model': 32,
    'large_model': 16
}
```

### 5. Monitoring Training

```python
import matplotlib.pyplot as plt

# After training
plt.figure(figsize=(10, 5))
plt.plot(history['train_losses'], label='Train Loss')
plt.plot(history['val_losses'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
```

### 6. Evaluating Results

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get predictions
predictions = trainer.predict(val_loader)

# Denormalize
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actuals = scaler.inverse_transform(val_data[lookback:].reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(actuals[:len(predictions)], predictions)
mae = mean_absolute_error(actuals[:len(predictions)], predictions)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
```

## Common Issues and Solutions

### Issue: Model not learning (loss not decreasing)

**Solutions**:
1. Check data normalization
2. Reduce learning rate (try 1e-5)
3. Increase model size
4. Check for data leakage

### Issue: Overfitting (train loss << val loss)

**Solutions**:
1. Increase dropout (0.2-0.3)
2. Add weight decay
3. Reduce model size
4. Get more training data
5. Use early stopping

### Issue: GPU out of memory

**Solutions**:
1. Reduce batch size
2. Reduce model size (`d_model`, `d_ff`)
3. Reduce sequence length

### Issue: Training too slow

**Solutions**:
1. Increase batch size
2. Reduce model size
3. Use mixed precision training
4. Increase stride in dataset

## Example: Training on Stock Price Data

```python
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Download stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y", interval="1d")

# Use closing prices
data = df['Close'].values.reshape(-1, 1)

# Normalize
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Split
train_size = int(0.8 * len(data_normalized))
train_data = data_normalized[:train_size]
val_data = data_normalized[train_size:]

# Create dataset with appropriate parameters
train_dataset = TimeSeriesDataset(
    train_data,
    lookback=60,        # Use 60 days to predict
    forecast_horizon=5,  # Predict 5 days ahead
    stride=1
)

# Continue with standard training workflow...
```

## Next Steps

- See `examples/basic_usage.py` for complete working example
- See `examples/multivariate_example.py` for multi-feature forecasting
- Check `QUICKSTART.md` for quick reference
- Read `ARCHITECTURE.md` for model details
