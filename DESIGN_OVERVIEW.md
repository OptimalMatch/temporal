# Temporal: Design Overview

## Architecture Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                      TEMPORAL MODEL                              │
│                   Time Series Forecasting                        │
└─────────────────────────────────────────────────────────────────┘

INPUT: Historical Time Series
  Shape: (batch, lookback, features)
  Example: (32, 96, 1) - 32 samples, 96 time steps, 1 feature
                    │
                    ▼
         ┌──────────────────────┐
         │  INPUT EMBEDDING     │
         │  Linear: f → d_model │
         │  (1 → 512)           │
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ POSITIONAL ENCODING  │
         │ sin/cos functions    │
         │ or learned embeddings│
         └──────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│                        ENCODER STACK                            │
│                         (6 layers)                              │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Layer 1                                                 │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Multi-Head Self-Attention (8 heads)              │  │   │
│  │  │ Q, K, V = Linear(x)                              │  │   │
│  │  │ Attention(Q,K,V) = softmax(QK^T/√d_k)V          │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Residual + LayerNorm                             │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Feed-Forward Network                             │  │   │
│  │  │ FFN(x) = GELU(xW₁ + b₁)W₂ + b₂                 │  │   │
│  │  │ d_model → d_ff → d_model (512→2048→512)        │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Residual + LayerNorm                             │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Layers 2-6 repeat same structure]                            │
│                                                                 │
│  Final LayerNorm                                               │
└────────────────────────────────────────────────────────────────┘
                    │
                    │ Encoder Output
                    │ Shape: (batch, lookback, d_model)
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│                        DECODER STACK                            │
│                         (6 layers)                              │
│                                                                 │
│  Decoder Input: Previous predictions (autoregressive)          │
│  or Target values (teacher forcing)                            │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Layer 1                                                 │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Masked Multi-Head Self-Attention                 │  │   │
│  │  │ Prevents attending to future positions           │  │   │
│  │  │ Uses causal mask                                 │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Residual + LayerNorm                             │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Cross-Attention with Encoder Output              │  │   │
│  │  │ Q from decoder, K,V from encoder                 │  │   │
│  │  │ Attends to input sequence                        │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Residual + LayerNorm                             │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Feed-Forward Network                             │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                      ↓                                  │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Residual + LayerNorm                             │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Layers 2-6 repeat same structure]                            │
│                                                                 │
│  Final LayerNorm                                               │
└────────────────────────────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  OUTPUT PROJECTION   │
         │  Linear: d_model → f │
         │  (512 → 1)           │
         └──────────────────────┘
                    │
                    ▼
OUTPUT: Forecast
  Shape: (batch, horizon, features)
  Example: (32, 24, 1) - 32 samples, 24 future steps, 1 feature
```

## Attention Mechanism Detail

```
Multi-Head Attention (8 heads)

Input: x (batch, seq_len, d_model=512)
       │
       ├─────────┬─────────┬─────────┐
       ▼         ▼         ▼         ▼
     Linear    Linear    Linear
     (Q)       (K)       (V)
     512→512   512→512   512→512
       │         │         │
       ▼         ▼         ▼
    Split into 8 heads (64 dims each)
       │         │         │
       └─────────┴─────────┘
              │
              ▼
      ┌──────────────────┐
      │  Head 1: 64 dims │  }
      │  Head 2: 64 dims │  }
      │  Head 3: 64 dims │  } Parallel
      │  ...             │  } Computation
      │  Head 8: 64 dims │  }
      └──────────────────┘
              │
              ▼
    Scaled Dot-Product Attention
    scores = QK^T / √64
    weights = softmax(scores)
    output = weights × V
              │
              ▼
    Concatenate all heads
    (8 × 64 = 512 dims)
              │
              ▼
    Final Linear Projection
              │
              ▼
    Output (batch, seq_len, 512)
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   Raw Time Series
        │
        ▼
   Normalize (StandardScaler)
        │
        ▼
   Create Sliding Windows
   [0:96] → [96:120]
   [1:97] → [97:121]
   ...
        │
        ▼
   DataLoader (batch_size=32)

2. FORWARD PASS (Teacher Forcing)
   Source: historical[0:96]
   Target: future[96:120]
        │
        ▼
   Encoder(source) → encoder_output
        │
        ▼
   Decoder(target, encoder_output) → predictions
        │
        ▼
   Loss = MSE(predictions, target)

3. BACKWARD PASS
   Loss.backward()
        │
        ▼
   Gradient Clipping (max_norm=1.0)
        │
        ▼
   Optimizer.step() (AdamW)
        │
        ▼
   Update weights

4. VALIDATION
   For each batch:
     predictions = model.forecast(source)
     val_loss = MSE(predictions, target)
        │
        ▼
   Average validation loss
        │
        ▼
   Early stopping check

5. CHECKPOINTING
   If val_loss < best_loss:
     Save model weights
     best_loss = val_loss
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  AUTOREGRESSIVE GENERATION                   │
└─────────────────────────────────────────────────────────────┘

Input: Historical data [0:96]

Step 1:
  Encode historical → encoder_output
  decoder_input = last_value[95]
  prediction[0] = decode(decoder_input, encoder_output)

Step 2:
  decoder_input = [95] + prediction[0]
  prediction[1] = decode(decoder_input, encoder_output)

Step 3:
  decoder_input = [95] + prediction[0:1]
  prediction[2] = decode(decoder_input, encoder_output)

...

Step 24:
  decoder_input = [95] + prediction[0:23]
  prediction[23] = decode(decoder_input, encoder_output)

Final Output: prediction[0:24]
```

## Data Flow Example

```
Example: Stock price forecasting

Input Data:
  Historical prices: [100, 102, 101, 103, ..., 105]
  Length: 96 days
  Features: 1 (closing price)

Normalization:
  mean = 102.5, std = 5.2
  normalized = (price - 102.5) / 5.2

Model Input:
  Shape: (1, 96, 1)
  Values: normalized prices

Model Processing:
  Embedding: 1 → 512 dims
  Encoder: Extract patterns
  Decoder: Generate forecast
  Projection: 512 → 1 dim

Model Output:
  Shape: (1, 24, 1)
  Values: normalized predictions

Denormalization:
  predictions = predictions * 5.2 + 102.5

Final Forecast:
  Next 24 days: [106, 107, 106, 108, ...]
```

## Module Dependencies

```
temporal/
│
├── attention.py
│   └── MultiHeadAttention
│       ├── Linear layers (Q, K, V, O)
│       ├── Scaled dot-product attention
│       └── Multi-head mechanism
│
├── position_encoding.py
│   ├── TemporalPositionEncoding (sinusoidal)
│   └── LearnablePositionEncoding (learned)
│
├── encoder.py
│   ├── FeedForward
│   │   └── Linear + GELU + Linear
│   ├── EncoderLayer
│   │   ├── MultiHeadAttention
│   │   ├── FeedForward
│   │   └── LayerNorm × 2
│   └── Encoder
│       └── Stack of EncoderLayer
│
├── decoder.py
│   ├── DecoderLayer
│   │   ├── MultiHeadAttention (self)
│   │   ├── MultiHeadAttention (cross)
│   │   ├── FeedForward
│   │   └── LayerNorm × 3
│   └── Decoder
│       └── Stack of DecoderLayer
│
├── model.py
│   └── Temporal
│       ├── Input embedding
│       ├── Positional encoding
│       ├── Encoder
│       ├── Decoder
│       └── Output projection
│
├── trainer.py
│   ├── TimeSeriesDataset
│   └── TemporalTrainer
│
└── utils.py
    ├── normalize_data
    ├── calculate_metrics
    ├── EarlyStopping
    └── LearningRateScheduler
```

## Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Flexibility**: Configurable architecture for different use cases
3. **Efficiency**: Parallel computation where possible
4. **Stability**: Residual connections and layer normalization
5. **Interpretability**: Attention weights can be visualized
6. **Extensibility**: Easy to add new features or modify components

## Performance Optimizations

- **Batch Processing**: Process multiple sequences in parallel
- **GPU Acceleration**: All operations are GPU-compatible
- **Gradient Checkpointing**: Option to trade compute for memory
- **Mixed Precision**: Support for FP16 training
- **Efficient Attention**: O(n²) but highly parallelizable

## Summary

Temporal is a complete, well-architected implementation of a transformer-based time series forecasting model. The design emphasizes modularity, flexibility, and production-readiness while maintaining the core principles of the original Transformer architecture, adapted for time series data.
