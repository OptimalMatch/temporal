"""
Unit tests for Temporal model.
"""

import torch
import pytest
import sys
sys.path.append('..')

from temporal import Temporal
from temporal.attention import MultiHeadAttention
from temporal.encoder import Encoder
from temporal.decoder import Decoder
from temporal.position_encoding import TemporalPositionEncoding


def test_multihead_attention():
    """Test multi-head attention mechanism."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    attention = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = attention(x, x, x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_positional_encoding():
    """Test positional encoding."""
    batch_size = 2
    seq_len = 100
    d_model = 64

    pe = TemporalPositionEncoding(d_model)

    x = torch.randn(batch_size, seq_len, d_model)
    output = pe(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_encoder():
    """Test encoder."""
    batch_size = 2
    seq_len = 50
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 256

    encoder = Encoder(num_layers, d_model, num_heads, d_ff)

    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_decoder():
    """Test decoder."""
    batch_size = 2
    src_len = 50
    tgt_len = 20
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 256

    decoder = Decoder(num_layers, d_model, num_heads, d_ff)

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    output = decoder(x, encoder_output)

    assert output.shape == (batch_size, tgt_len, d_model)


def test_temporal_model_forward():
    """Test Temporal model forward pass."""
    batch_size = 4
    src_len = 96
    tgt_len = 24
    input_dim = 1

    model = Temporal(
        input_dim=input_dim,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        forecast_horizon=tgt_len
    )

    src = torch.randn(batch_size, src_len, input_dim)
    tgt = torch.randn(batch_size, tgt_len, input_dim)

    output = model(src, tgt)

    assert output.shape == (batch_size, tgt_len, input_dim)


def test_temporal_model_forecast():
    """Test Temporal model forecasting (autoregressive)."""
    batch_size = 4
    src_len = 96
    forecast_horizon = 24
    input_dim = 1

    model = Temporal(
        input_dim=input_dim,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        forecast_horizon=forecast_horizon
    )

    src = torch.randn(batch_size, src_len, input_dim)
    forecast = model.forecast(src)

    assert forecast.shape == (batch_size, forecast_horizon, input_dim)


def test_temporal_multivariate():
    """Test Temporal with multivariate time series."""
    batch_size = 4
    src_len = 96
    forecast_horizon = 24
    input_dim = 5  # 5 features

    model = Temporal(
        input_dim=input_dim,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        forecast_horizon=forecast_horizon
    )

    src = torch.randn(batch_size, src_len, input_dim)
    forecast = model.forecast(src)

    assert forecast.shape == (batch_size, forecast_horizon, input_dim)


def test_causal_mask():
    """Test causal mask generation."""
    model = Temporal(
        input_dim=1,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        forecast_horizon=24
    )

    size = 10
    mask = model.generate_causal_mask(size)

    assert mask.shape == (size, size)
    # Check that mask prevents looking ahead
    assert mask[0, 1] == False  # Can't look at future
    assert mask[5, 3] == True   # Can look at past


def test_model_parameters():
    """Test that model has expected number of parameters."""
    model = Temporal(
        input_dim=1,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        forecast_horizon=24
    )

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    # Run tests
    test_multihead_attention()
    print("✓ Multi-head attention test passed")

    test_positional_encoding()
    print("✓ Positional encoding test passed")

    test_encoder()
    print("✓ Encoder test passed")

    test_decoder()
    print("✓ Decoder test passed")

    test_temporal_model_forward()
    print("✓ Temporal forward test passed")

    test_temporal_model_forecast()
    print("✓ Temporal forecast test passed")

    test_temporal_multivariate()
    print("✓ Temporal multivariate test passed")

    test_causal_mask()
    print("✓ Causal mask test passed")

    test_model_parameters()
    print("✓ Model parameters test passed")

    print("\n✅ All tests passed!")
