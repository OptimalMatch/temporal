# Temporal Quick Start Guide

Get started with Temporal in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/OptimalMatch/temporal.git
cd temporal

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Basic Example (5 lines of code)

```python
import torch
from temporal import Temporal

# Create model
model = Temporal(input_dim=1, forecast_horizon=24)

# Make forecast
x = torch.randn(1, 96, 1)  # 96 historical points
forecast = model.forecast(x)  # Predict next 24 points

print(forecast.shape)  # torch.Size([1, 24, 1])
```

## Complete Training Example

### 1. Prepare Your Data

```python
import numpy as np
from temporal import normalize_data, split_train_val_test

# Load your time series data
data = np.load('your_data.npy')  # Shape: (num_samples, num_features)

# Normalize
data, norm_params = normalize_data(data, method='standard')

# Split
train_data, val_data, test_data = split_train_val_test(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 2. Create Dataset and DataLoader

```python
from temporal import TimeSeriesDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = TimeSeriesDataset(
    train_data,
    lookback=96,        # Use 96 past points
    forecast_horizon=24, # Predict 24 future points
    stride=1
)

# Create dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### 3. Create and Train Model

```python
import torch
from temporal import Temporal, TemporalTrainer

# Create model
model = Temporal(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    forecast_horizon=24
)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create trainer
trainer = TemporalTrainer(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.MSELoss()
)

# Train!
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10,
    save_path='best_model.pt'
)
```

### 4. Make Predictions

```python
# Generate predictions
predictions = trainer.predict(test_loader)

# Calculate metrics
from temporal import calculate_metrics, denormalize_data

# Denormalize predictions
predictions = denormalize_data(predictions, norm_params)
actual = denormalize_data(test_data, norm_params)

# Calculate metrics
metrics = calculate_metrics(predictions, actual[:len(predictions)])
print(metrics)
```

## Run Example Scripts

### Univariate Forecasting

```bash
cd examples
python basic_usage.py
```

This will:
- Generate synthetic time series data
- Train a Temporal model
- Make forecasts
- Save visualizations

### Multivariate Forecasting

```bash
cd examples
python multivariate_example.py
```

This demonstrates forecasting multiple correlated features.

## Common Use Cases

### Univariate Time Series (e.g., Stock Price)

```python
model = Temporal(
    input_dim=1,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    forecast_horizon=30  # Predict 30 days ahead
)
```

### Multivariate Time Series (e.g., Weather Data)

```python
model = Temporal(
    input_dim=5,  # Temperature, humidity, pressure, wind speed, rainfall
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    forecast_horizon=24  # Predict 24 hours ahead
)
```

### Short Sequences (Fast Inference)

```python
model = Temporal(
    input_dim=1,
    d_model=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_ff=512,
    forecast_horizon=12
)
```

### Long Sequences (High Accuracy)

```python
model = Temporal(
    input_dim=1,
    d_model=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=16,
    d_ff=2048,
    forecast_horizon=96
)
```

## Tips for Best Results

1. **Normalize Your Data**: Always normalize time series data
   ```python
   data, params = normalize_data(data, method='standard')
   ```

2. **Use Appropriate Lookback**: Generally 2-5x the forecast horizon
   ```python
   lookback = 4 * forecast_horizon
   ```

3. **Start Small**: Begin with a small model and increase size if needed
   ```python
   # Start with this
   model = Temporal(d_model=128, num_encoder_layers=2, ...)
   ```

4. **Monitor Validation Loss**: Use early stopping
   ```python
   trainer.fit(..., early_stopping_patience=10)
   ```

5. **Use GPU if Available**: Much faster training
   ```python
   trainer = TemporalTrainer(..., device='cuda')
   ```

## Troubleshooting

### Out of Memory Error
- Reduce batch size
- Reduce model size (d_model, num_layers)
- Use gradient checkpointing

### Poor Accuracy
- Increase model size
- Train for more epochs
- Adjust learning rate
- Check data normalization
- Increase lookback window

### Slow Training
- Use GPU
- Increase batch size
- Reduce sequence length
- Use smaller model for prototyping

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details
- See [DIAGRAMS.md](DIAGRAMS.md) for visual architecture diagrams
- Explore example scripts in `examples/`
- Run tests: `cd tests && python test_model.py`

## Support

For issues and questions:
- Documentation: See README.md and ARCHITECTURE.md
- Examples: Check the examples/ directory
