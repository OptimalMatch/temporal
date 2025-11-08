# Temporal: Transformer-Based Time Series Forecasting

A PyTorch implementation of a transformer-based model for time series forecasting, inspired by modern attention-based approaches.

## Overview

Temporal is a foundational model for time series forecasting based on the revolutionary self-attention mechanism introduced in "Attention is All You Need". Unlike language models, Temporal is specifically designed and trained to minimize forecasting error on time series data.

### Key Features

- **Self-Attention Mechanism**: Captures complex temporal dependencies and patterns
- **Encoder-Decoder Architecture**: Multi-layer transformer with residual connections and layer normalization
- **Flexible**: Supports both univariate and multivariate time series
- **Scalable**: Can handle various sequence lengths and forecasting horizons
- **Autoregressive Generation**: Inference mode for multi-step ahead forecasting

## Architecture

The Temporal model consists of:

1. **Input Embedding**: Projects time series data to model dimension
2. **Positional Encoding**: Captures temporal order (sinusoidal or learnable)
3. **Encoder Stack**: Multiple layers of self-attention and feed-forward networks
4. **Decoder Stack**: Multiple layers with self-attention, cross-attention, and feed-forward networks
5. **Output Projection**: Maps decoder output to forecasting window dimension

### Architecture Diagram

```mermaid
graph TD
    A[Input Time Series<br/>batch, lookback, features] --> B[Input Embedding<br/>Linear: features → d_model]
    B --> C[Positional Encoding<br/>Add temporal position info]
    C --> D[Encoder Stack<br/>6 layers]
    D --> E[Encoder Output<br/>batch, lookback, d_model]

    F[Decoder Input<br/>Previous predictions] --> G[Input Embedding<br/>Linear: features → d_model]
    G --> H[Positional Encoding]
    H --> I[Decoder Stack<br/>6 layers]
    E --> I
    I --> J[Decoder Output<br/>batch, horizon, d_model]
    J --> K[Output Projection<br/>Linear: d_model → features]
    K --> L[Forecast<br/>batch, horizon, features]

    style A fill:#e1f5ff
    style L fill:#e1ffe1
    style D fill:#fff4e1
    style I fill:#ffe1f5
```

Each layer includes:
- Multi-head self-attention
- Residual connections
- Layer normalization
- Feed-forward networks with GELU activation

**For more diagrams**, see [DIAGRAMS.md](DIAGRAMS.md) - complete visual documentation with:
- Encoder/Decoder architecture
- Multi-head attention mechanism
- Training and inference flows
- Data pipeline
- Component interactions

## Installation

### From PyPI

```bash
pip install temporal-forecasting
```

### With HuggingFace Support

```bash
pip install temporal-forecasting[huggingface]
```

This adds support for:
- Uploading models to HuggingFace Hub
- Downloading models from HuggingFace Hub
- HuggingFace ecosystem integration

### With Data Fetching Support

```bash
pip install temporal-forecasting[data]
```

This adds support for:
- Fetching stock prices from Yahoo Finance
- Fetching cryptocurrency data (Bitcoin, Ethereum, etc.)
- Downloading datasets from Kaggle
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Data preprocessing utilities

### From Source

```bash
git clone https://github.com/OptimalMatch/temporal.git
cd temporal
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- tqdm >= 4.60.0
- matplotlib >= 3.3.0

### Optional Dependencies

- **HuggingFace**: `transformers>=4.30.0`, `huggingface-hub>=0.16.0`
- **Data Fetching**: `yfinance>=0.2.0`, `pandas>=1.3.0`, `scikit-learn>=1.0.0`, `kagglehub>=0.2.0`

## Quick Start

### Basic Usage

```python
import torch
from temporal import Temporal

# Create model
model = Temporal(
    input_dim=1,           # Univariate time series
    d_model=256,           # Model dimension
    num_encoder_layers=4,  # Number of encoder layers
    num_decoder_layers=4,  # Number of decoder layers
    num_heads=8,           # Attention heads
    d_ff=1024,            # Feed-forward dimension
    forecast_horizon=24,   # Predict 24 steps ahead
    dropout=0.1
)

# Input: (batch_size, sequence_length, input_dim)
x = torch.randn(32, 96, 1)

# Generate forecast
forecast = model.forecast(x)  # (32, 24, 1)
```

### Training Example

```python
from temporal import Temporal
from temporal.trainer import TimeSeriesDataset, TemporalTrainer
from torch.utils.data import DataLoader
import torch

# Prepare your data
train_data = ...  # Shape: (num_samples, num_features)

# Create dataset
dataset = TimeSeriesDataset(
    train_data,
    lookback=96,
    forecast_horizon=24,
    stride=1
)

# Create data loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = Temporal(
    input_dim=train_data.shape[1],
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024,
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

# Train
history = trainer.fit(
    train_loader=train_loader,
    num_epochs=100,
    early_stopping_patience=10,
    save_path="best_model.pt"  # Automatically saves best model
)
```

### Saving and Loading Models

```python
# Save trained model
torch.save(model.state_dict(), 'temporal_model.pt')

# Load model for inference
model = Temporal(input_dim=1, forecast_horizon=24)
model.load_state_dict(torch.load('temporal_model.pt'))
model.eval()

# Make predictions
forecast = model.forecast(x)
```

For complete guide on model persistence, see [MODEL_PERSISTENCE.md](MODEL_PERSISTENCE.md).

## Examples

### Univariate Time Series

See `examples/basic_usage.py` for a complete example with synthetic data:

```bash
cd examples
python basic_usage.py
```

This will:
- Generate synthetic time series data
- Train a Temporal model
- Generate forecasts
- Visualize results

### Multivariate Time Series

See `examples/multivariate_example.py` for forecasting multiple correlated features:

```bash
cd examples
python multivariate_example.py
```

### Model Persistence

See `examples/model_persistence_example.py` for saving and loading trained models:

```bash
cd examples
python model_persistence_example.py
```

This demonstrates:
- Training and saving a model with all components
- Loading saved models for inference
- Production-ready model deployment

### HuggingFace Integration

See `examples/huggingface_example.py` for HuggingFace Hub integration:

```bash
cd examples
python huggingface_example.py
```

This demonstrates:
- Creating HuggingFace-compatible models
- Saving in HuggingFace format
- Loading from HuggingFace Hub
- Uploading models to share with the community

For complete guide, see [HUGGINGFACE_INTEGRATION.md](HUGGINGFACE_INTEGRATION.md).

### Stock Price Forecasting

See `examples/stock_forecasting.py` for real stock data forecasting:

```bash
cd examples
python stock_forecasting.py
```

This demonstrates:
- Fetching stock data from Yahoo Finance
- Training on Apple (AAPL) stock prices
- 5-day price forecasting
- Model evaluation and visualization

### Cryptocurrency Forecasting

See `examples/crypto_forecasting.py` for Bitcoin and crypto forecasting:

```bash
cd examples
python crypto_forecasting.py
```

This demonstrates:
- Fetching Bitcoin data
- Training on cryptocurrency prices
- 7-day price forecasting
- Multi-crypto comparison

For complete guide on data fetching, see [DATA_SOURCES.md](DATA_SOURCES.md).

## Reference Implementations

The following projects demonstrate real-world applications built using the Temporal forecasting library. These implementations showcase how to integrate Temporal into production systems and can serve as templates for your own projects.

### Temporal Trading Agents

**Repository:** [github.com/OptimalMatch/temporal-trading-agents](https://github.com/OptimalMatch/temporal-trading-agents)

A next-generation trading system that combines deep learning time-series forecasting with ensemble methods and multi-strategy consensus voting to predict market movements and generate trading signals.

#### Features

- **Multi-Horizon Forecasting**: Separate ensembles for 3-day, 7-day, 14-day, and 21-day predictions
- **Ensemble Learning**: Combines 5-8 models per time horizon with confidence quantification
- **8-Strategy Consensus System**: Analyzes predictions using gradient analysis, confidence weighting, volatility sizing, momentum, swing trading, risk-adjusted metrics, mean reversion, and multi-timeframe alignment
- **Production-Ready Platform**: React dashboard with FastAPI backend, MongoDB, and Docker deployment
- **Risk Management**: Dynamic position sizing, VaR calculations, and Sortino ratio analysis

#### Using Temporal in Your Project

Add Temporal to your `requirements.txt`:

```txt
temporal-forecasting>=0.3.1
```

Example usage from the trading agents implementation:

```python
from temporal import Temporal, TemporalTrainer, TimeSeriesDataset
from temporal.data_sources import fetch_crypto_data
import torch

# Fetch cryptocurrency data
data = fetch_crypto_data('BTC-USD', period='2y')

# Create and train ensemble of models for different horizons
horizons = [3, 7, 14, 21]  # days
models = {}

for horizon in horizons:
    # Prepare dataset
    dataset = TimeSeriesDataset(
        data,
        lookback=96,
        forecast_horizon=horizon * 24,  # Convert days to hours
        stride=1
    )

    # Create model
    model = Temporal(
        input_dim=data.shape[1],
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        forecast_horizon=horizon * 24
    )

    # Train model
    trainer = TemporalTrainer(model, optimizer=torch.optim.AdamW(model.parameters()))
    history = trainer.fit(train_loader, num_epochs=100)

    models[f'{horizon}d'] = model

# Generate multi-horizon forecasts
forecasts = {}
for horizon, model in models.items():
    forecast = model.forecast(recent_data)
    forecasts[horizon] = forecast

# Use forecasts for trading strategy consensus voting
# (See temporal-trading-agents for full strategy implementation)
```

#### Learn More

- **Documentation**: See the [temporal-trading-agents README](https://github.com/OptimalMatch/temporal-trading-agents#readme)
- **Live Demo**: Follow the Docker setup instructions for a complete trading dashboard
- **Strategies**: Review the 8-strategy consensus voting system for signal generation

---

### Contributing Your Implementation

Have you built something with Temporal? We'd love to feature your project! Submit a pull request adding your implementation to this section, including:

- Project description and repository link
- Key features and use cases
- Code example showing Temporal integration
- Any unique approaches or optimizations

## Model Configuration

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dim` | Number of input features | 1 |
| `d_model` | Model dimension | 512 |
| `num_encoder_layers` | Number of encoder layers | 6 |
| `num_decoder_layers` | Number of decoder layers | 6 |
| `num_heads` | Number of attention heads | 8 |
| `d_ff` | Feed-forward dimension | 2048 |
| `forecast_horizon` | Number of steps to forecast | 24 |
| `max_seq_len` | Maximum sequence length | 5000 |
| `dropout` | Dropout probability | 0.1 |
| `use_learnable_pe` | Use learnable positional encoding | False |

### Recommended Configurations

**Small Model** (Fast training, lower accuracy):
```python
model = Temporal(
    d_model=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_ff=512
)
```

**Medium Model** (Balanced):
```python
model = Temporal(
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=8,
    d_ff=1024
)
```

**Large Model** (Best accuracy, slower training):
```python
model = Temporal(
    d_model=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=16,
    d_ff=2048
)
```

## Training Tips

1. **Learning Rate**: Start with 1e-4 and use a scheduler (e.g., ReduceLROnPlateau)
2. **Batch Size**: Use the largest batch size that fits in memory (32-128)
3. **Gradient Clipping**: Use gradient clipping (0.5-1.0) to prevent exploding gradients
4. **Early Stopping**: Monitor validation loss and stop when it plateaus
5. **Data Normalization**: Normalize your data (e.g., StandardScaler) before training

## Architecture Details

### Multi-Head Attention

The model uses scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Multiple attention heads allow the model to attend to different aspects of the time series simultaneously.

### Positional Encoding

Two types of positional encoding are available:

1. **Sinusoidal** (default): Fixed sinusoidal functions
2. **Learnable**: Learned embeddings for each position

### Autoregressive Generation

During inference, the model generates forecasts autoregressively:
- Start with the last observed value
- Generate next step prediction
- Use prediction as input for next step
- Repeat for entire forecast horizon

## Modern Time Series Transformers

Temporal implements a transformer architecture similar to modern approaches in time series forecasting:

| Feature | Modern Approaches | Temporal |
|---------|---------|----------|
| Architecture | Transformer | Transformer |
| Attention | Multi-head | Multi-head |
| Layers | Encoder-Decoder | Encoder-Decoder |
| Training | Large-scale pre-training | User-provided data |
| Flexibility | Fixed models | Fully customizable |

## Performance

Performance varies by dataset and configuration. Typical metrics on benchmark datasets:

- **MSE**: 0.01-0.1 (normalized data)
- **MAE**: 0.05-0.3 (normalized data)
- **Training Time**: 1-10 minutes per epoch (depending on size)

## API Reference

### Temporal

Main model class for time series forecasting.

**Methods**:
- `forward(src, tgt=None, src_mask=None, tgt_mask=None)`: Forward pass
- `forecast(x, horizon=None)`: Generate forecasts
- `generate_causal_mask(size)`: Create causal attention mask

### TemporalTrainer

Training utilities for Temporal models.

**Methods**:
- `train_epoch(dataloader)`: Train for one epoch
- `validate(dataloader)`: Validate the model
- `fit(train_loader, val_loader, num_epochs, ...)`: Full training loop
- `predict(dataloader)`: Generate predictions

### TimeSeriesDataset

Dataset class for time series data.

**Parameters**:
- `data`: Time series data array
- `lookback`: Number of historical steps
- `forecast_horizon`: Number of future steps
- `stride`: Stride for sliding window

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Important:** By contributing to this project, you agree to the terms of the [Contributor Assignment Agreement (CAA)](CONTRIBUTOR_AGREEMENT.md), which assigns copyright of your contributions to Unidatum Integrated Products LLC. Please include the CAA statement in your pull request.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3) - see the [LICENSE](LICENSE) file for details.

Copyright (C) 2025 Unidatum Integrated Products LLC

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

### Commercial Licensing

Companies that cannot release their source code under the GPLv3 may purchase a commercial license from Unidatum Integrated Products LLC. A commercial license grants the right to use this software in closed-source, proprietary projects without the requirement to disclose source code.

For commercial licensing inquiries, please contact: licensing@unidatum.com

## Patents

This software is subject to a pending patent application:

**Transformer-based Time Series Forecasting System and Method**
US Patent Application No. 63/910,189 (Filed November 3, 2025)

The patent covers specific methods and systems related to transformer-based time series forecasting. Use of this software under the GPLv3 license includes the patent license provisions specified in Section 11 of the GPLv3. For more information, see the [PATENTS](PATENTS) file.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{temporal2024,
  title = {Temporal: Transformer-Based Time Series Forecasting},
  year = {2024},
  note = {A PyTorch implementation of transformer architecture for time series},
  url = {https://github.com/OptimalMatch/temporal}
}
```

## References

- Vaswani et al., "Attention is All You Need" (2017)
- Modern transformer-based time series forecasting approaches

## Acknowledgments

This implementation is inspired by modern transformer architectures for time series forecasting and the original Transformer paper.
