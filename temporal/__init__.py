"""
Temporal: A transformer-based time series forecasting model.

This model uses self-attention mechanisms to capture temporal dependencies
in time series data, inspired by modern transformer architectures.
"""

from .model import Temporal
from .attention import MultiHeadAttention
from .encoder import Encoder, EncoderLayer, FeedForward
from .decoder import Decoder, DecoderLayer
from .position_encoding import TemporalPositionEncoding, LearnablePositionEncoding
from .trainer import TemporalTrainer, TimeSeriesDataset
from .utils import (
    normalize_data,
    denormalize_data,
    split_train_val_test,
    calculate_metrics,
    count_parameters,
    EarlyStopping,
    LearningRateScheduler
)

__version__ = "0.1.0"

__all__ = [
    "Temporal",
    "MultiHeadAttention",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "FeedForward",
    "TemporalPositionEncoding",
    "LearnablePositionEncoding",
    "TemporalTrainer",
    "TimeSeriesDataset",
    "normalize_data",
    "denormalize_data",
    "split_train_val_test",
    "calculate_metrics",
    "count_parameters",
    "EarlyStopping",
    "LearningRateScheduler",
]
