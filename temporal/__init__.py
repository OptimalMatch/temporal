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

__version__ = "0.3.0"

# Try to import HuggingFace interfaces (optional)
try:
    from .hf_interface import (
        TemporalConfig,
        TemporalForForecasting,
        TemporalHubMixin,
        create_hf_model
    )
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

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

# Add HuggingFace interfaces if available
if _HF_AVAILABLE:
    __all__.extend([
        "TemporalConfig",
        "TemporalForForecasting",
        "TemporalHubMixin",
        "create_hf_model",
    ])
