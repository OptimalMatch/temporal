# Temporal Architecture Diagrams

This document contains visual diagrams of the Temporal model architecture using Mermaid.

## Table of Contents
- [Overall Architecture](#overall-architecture)
- [Encoder Architecture](#encoder-architecture)
- [Decoder Architecture](#decoder-architecture)
- [Multi-Head Attention](#multi-head-attention)
- [Training Flow](#training-flow)
- [Inference Flow](#inference-flow)
- [Data Pipeline](#data-pipeline)

---

## Overall Architecture

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

---

## Encoder Architecture

```mermaid
graph TD
    A[Encoder Input<br/>batch, seq_len, d_model] --> B[Encoder Layer 1]
    B --> C[Encoder Layer 2]
    C --> D[...]
    D --> E[Encoder Layer 6]
    E --> F[Layer Normalization]
    F --> G[Encoder Output<br/>batch, seq_len, d_model]

    subgraph "Encoder Layer Structure"
        B1[Input] --> B2[Multi-Head Self-Attention]
        B2 --> B3[Add & Norm<br/>Residual + LayerNorm]
        B3 --> B4[Feed-Forward Network<br/>d_model → d_ff → d_model]
        B4 --> B5[Add & Norm<br/>Residual + LayerNorm]
        B5 --> B6[Output]
    end

    style A fill:#e1f5ff
    style G fill:#e1ffe1
    style B2 fill:#fff4e1
    style B4 fill:#ffe1f5
```

---

## Decoder Architecture

```mermaid
graph TD
    A[Decoder Input<br/>batch, tgt_len, d_model] --> B[Decoder Layer 1]
    B --> C[Decoder Layer 2]
    C --> D[...]
    D --> E[Decoder Layer 6]
    E --> F[Layer Normalization]
    F --> G[Decoder Output<br/>batch, tgt_len, d_model]

    H[Encoder Output] -.-> B
    H -.-> C
    H -.-> E

    subgraph "Decoder Layer Structure"
        D1[Input] --> D2[Masked Self-Attention<br/>with causal mask]
        D2 --> D3[Add & Norm]
        D3 --> D4[Cross-Attention<br/>Q: decoder, K,V: encoder]
        D4 --> D5[Add & Norm]
        D5 --> D6[Feed-Forward Network]
        D6 --> D7[Add & Norm]
        D7 --> D8[Output]
    end

    style A fill:#e1f5ff
    style G fill:#e1ffe1
    style D2 fill:#fff4e1
    style D4 fill:#ffe1f5
    style D6 fill:#e1f0ff
```

---

## Multi-Head Attention

```mermaid
graph TD
    A[Input X<br/>batch, seq_len, d_model] --> B[Linear Q]
    A --> C[Linear K]
    A --> D[Linear V]

    B --> E[Split into h heads<br/>batch, h, seq_len, d_k]
    C --> F[Split into h heads<br/>batch, h, seq_len, d_k]
    D --> G[Split into h heads<br/>batch, h, seq_len, d_k]

    E --> H[Head 1: Attention]
    E --> I[Head 2: Attention]
    E --> J[...]
    E --> K[Head h: Attention]

    F --> H
    F --> I
    F --> J
    F --> K

    G --> H
    G --> I
    G --> J
    G --> K

    H --> L[Concatenate Heads]
    I --> L
    J --> L
    K --> L

    L --> M[Linear Output<br/>d_model → d_model]
    M --> N[Output<br/>batch, seq_len, d_model]

    subgraph "Scaled Dot-Product Attention"
        H1[Q, K, V] --> H2[MatMul: Q × K^T]
        H2 --> H3[Scale: ÷ √d_k]
        H3 --> H4[Optional Mask]
        H4 --> H5[Softmax]
        H5 --> H6[MatMul: × V]
        H6 --> H7[Output]
    end

    style A fill:#e1f5ff
    style N fill:#e1ffe1
    style H fill:#fff4e1
    style I fill:#fff4e1
    style K fill:#fff4e1
```

---

## Training Flow

```mermaid
graph LR
    A[Raw Time Series] --> B[Normalize Data]
    B --> C[Create Sliding Windows<br/>lookback → horizon]
    C --> D[DataLoader<br/>batch_size=32]

    D --> E[Forward Pass]
    E --> E1[Encode Source]
    E1 --> E2[Decode with Teacher Forcing<br/>use actual target values]
    E2 --> E3[Compute Loss<br/>MSE predictions vs targets]

    E3 --> F[Backward Pass]
    F --> F1[loss.backward]
    F1 --> F2[Gradient Clipping<br/>max_norm=1.0]
    F2 --> F3[Optimizer Step<br/>AdamW]

    F3 --> G{Epoch Complete?}
    G -->|No| D
    G -->|Yes| H[Validate]

    H --> I[Compute Val Loss]
    I --> J{Early Stop?}
    J -->|No| K{More Epochs?}
    J -->|Yes| L[Training Complete]
    K -->|Yes| D
    K -->|No| L

    L --> M[Best Model Saved]

    style A fill:#e1f5ff
    style M fill:#e1ffe1
    style E2 fill:#fff4e1
    style F2 fill:#ffe1f5
```

---

## Inference Flow

```mermaid
graph TD
    A[Historical Data<br/>batch, lookback, features] --> B[Encode]
    B --> C[Encoder Output]

    D[Last Historical Point] --> E[Decoder Input]

    C --> F1[Decode Step 1]
    E --> F1
    F1 --> G1[Prediction 1]

    G1 --> H1[Append to Decoder Input]
    H1 --> F2[Decode Step 2]
    C --> F2
    F2 --> G2[Prediction 2]

    G2 --> H2[Append to Decoder Input]
    H2 --> F3[Decode Step 3]
    C --> F3
    F3 --> G3[Prediction 3]

    G3 --> I[...]

    I --> F24[Decode Step 24]
    C --> F24
    F24 --> G24[Prediction 24]

    G1 --> J[Concatenate All]
    G2 --> J
    G3 --> J
    G24 --> J

    J --> K[Final Forecast<br/>batch, 24, features]

    style A fill:#e1f5ff
    style K fill:#e1ffe1
    style F1 fill:#fff4e1
    style F2 fill:#fff4e1
    style F24 fill:#fff4e1
```

---

## Data Pipeline

```mermaid
graph TD
    A[Raw Time Series<br/>num_samples, features] --> B{Normalization}

    B -->|Standard| B1[μ=0, σ=1]
    B -->|MinMax| B2[range: 0-1]
    B -->|Robust| B3[IQR-based]

    B1 --> C[Normalized Data]
    B2 --> C
    B3 --> C

    C --> D[Train/Val/Test Split<br/>70% / 15% / 15%]

    D --> E[Train Data]
    D --> F[Val Data]
    D --> G[Test Data]

    E --> H[TimeSeriesDataset<br/>Sliding Window]

    H --> I[Sample 1:<br/>src, decoder_input, target]
    H --> J[Sample 2:<br/>src, decoder_input, target]
    H --> K[...]
    H --> L[Sample N:<br/>src, decoder_input, target]

    I --> M[DataLoader<br/>Batching + Shuffling]
    J --> M
    K --> M
    L --> M

    M --> N[Batched Data<br/>batch_size, seq_len, features]

    N --> O[Model Training]

    subgraph "Sliding Window Example"
        SW1[Window 1: 0-96 → 96-120]
        SW2[Window 2: 1-97 → 97-121]
        SW3[Window 3: 2-98 → 98-122]
    end

    style A fill:#e1f5ff
    style O fill:#e1ffe1
    style H fill:#fff4e1
    style M fill:#ffe1f5
```

---

## Component Interaction

```mermaid
graph TD
    subgraph "temporal Package"
        A[model.py<br/>Temporal] --> B[encoder.py<br/>Encoder]
        A --> C[decoder.py<br/>Decoder]

        B --> D[attention.py<br/>MultiHeadAttention]
        C --> D

        B --> E[position_encoding.py<br/>TemporalPositionEncoding]
        C --> E

        A --> F[Output Projection<br/>Linear Layer]
    end

    subgraph "Training Infrastructure"
        G[trainer.py<br/>TemporalTrainer] --> A
        H[trainer.py<br/>TimeSeriesDataset] --> G
        I[utils.py<br/>normalize_data] --> H
        J[utils.py<br/>calculate_metrics] --> G
    end

    subgraph "User Interface"
        K[User Code] --> A
        K --> G
        K --> H
        K --> I
    end

    style A fill:#e1f5ff
    style G fill:#fff4e1
    style K fill:#e1ffe1
```

---

## Model Size Comparison

```mermaid
graph LR
    subgraph "Small Model"
        S1[d_model: 128]
        S2[Layers: 2+2]
        S3[Heads: 4]
        S4[Params: ~927K]
        S5[Speed: 200+ it/s]
    end

    subgraph "Medium Model"
        M1[d_model: 256]
        M2[Layers: 4+4]
        M3[Heads: 8]
        M4[Params: ~10M]
        M5[Speed: 50-100 it/s]
    end

    subgraph "Large Model"
        L1[d_model: 512]
        L2[Layers: 6+6]
        L3[Heads: 16]
        L4[Params: ~50M]
        L5[Speed: 10-30 it/s]
    end

    S1 --> M1
    M1 --> L1

    style S1 fill:#e1ffe1
    style M1 fill:#fff4e1
    style L1 fill:#ffe1f5
```

---

## Use Cases Flow

```mermaid
graph TD
    A[Time Series Data] --> B{Use Case}

    B -->|Finance| C[Stock Prices<br/>Portfolio Optimization]
    B -->|Energy| D[Load Forecasting<br/>Renewable Prediction]
    B -->|Weather| E[Temperature<br/>Precipitation]
    B -->|Healthcare| F[Patient Monitoring<br/>Epidemic Forecasting]
    B -->|Retail| G[Demand Forecasting<br/>Inventory Optimization]
    B -->|IoT| H[Sensor Data<br/>Anomaly Detection]

    C --> I[Temporal Model]
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I

    I --> J[Forecasts]

    J --> K{Evaluation}
    K -->|Good| L[Deploy to Production]
    K -->|Poor| M[Tune Hyperparameters]

    M --> I

    style A fill:#e1f5ff
    style I fill:#fff4e1
    style L fill:#e1ffe1
```

---

## Notes

- All diagrams render automatically on GitHub
- Diagrams use Mermaid syntax (supported natively in GitHub Markdown)
- Color coding:
  - Light blue: Input data
  - Light green: Output/results
  - Light yellow: Processing steps
  - Light pink: Critical operations
  - Light purple: Alternative options

For more details, see:
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture documentation
- [DESIGN_OVERVIEW.md](DESIGN_OVERVIEW.md) - Visual text-based diagrams
- [README.md](README.md) - Complete usage guide
