"""
Hugging Face integration for Temporal model.

Copyright (C) 2025 Unidatum Integrated Products LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

This module provides HuggingFace-compatible interfaces for the Temporal model,
enabling easy upload/download from the Hugging Face Hub.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import json
import os

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from huggingface_hub import PyTorchModelHubMixin
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Create dummy classes if transformers not installed
    class PretrainedConfig:
        pass
    class PreTrainedModel(nn.Module):
        pass
    class PyTorchModelHubMixin:
        pass

from .model import Temporal


class TemporalConfig(PretrainedConfig):
    """
    Configuration class for Temporal model.

    Compatible with HuggingFace PretrainedConfig.
    """

    model_type = "temporal"

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        forecast_horizon: int = 24,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        use_learnable_pe: bool = False,
        **kwargs
    ):
        """
        Initialize Temporal configuration.

        Args:
            input_dim: Number of input features
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            forecast_horizon: Number of steps to forecast
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_learnable_pe: Use learnable positional encoding
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.forecast_horizon = forecast_horizon
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_learnable_pe = use_learnable_pe


class TemporalForForecasting(PreTrainedModel):
    """
    HuggingFace-compatible Temporal model for time series forecasting.

    This wrapper allows the Temporal model to be used with HuggingFace Hub.
    """

    config_class = TemporalConfig
    base_model_prefix = "temporal"

    def __init__(self, config: TemporalConfig):
        """
        Initialize HF-compatible Temporal model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Create the actual Temporal model
        self.temporal = Temporal(
            input_dim=config.input_dim,
            d_model=config.d_model,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            forecast_horizon=config.forecast_horizon,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_learnable_pe=config.use_learnable_pe
        )

        # Post init
        self.post_init()

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass.

        Args:
            src: Source sequence (batch_size, seq_len, input_dim)
            tgt: Target sequence for teacher forcing (optional)
            src_mask: Source attention mask (optional)
            tgt_mask: Target attention mask (optional)

        Returns:
            Model output
        """
        return self.temporal(src, tgt, src_mask, tgt_mask)

    def forecast(self, x: torch.Tensor, horizon: Optional[int] = None):
        """
        Generate autoregressive forecast.

        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            horizon: Forecast horizon (uses config if not specified)

        Returns:
            Forecast (batch_size, horizon, input_dim)
        """
        return self.temporal.forecast(x, horizon)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load pretrained model from HuggingFace Hub or local path.

        Args:
            pretrained_model_name_or_path: Model ID on HF Hub or local path

        Returns:
            Loaded model
        """
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save model to directory (HF format).

        Args:
            save_directory: Directory to save to
        """
        super().save_pretrained(save_directory, **kwargs)


class TemporalHubMixin(PyTorchModelHubMixin, nn.Module):
    """
    Alternative HuggingFace Hub integration using PyTorchModelHubMixin.

    This is a simpler approach that doesn't require full transformers integration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with config dict.

        Args:
            config: Model configuration dictionary
        """
        nn.Module.__init__(self)

        self.config = config
        self.temporal = Temporal(
            input_dim=config.get('input_dim', 1),
            d_model=config.get('d_model', 512),
            num_encoder_layers=config.get('num_encoder_layers', 6),
            num_decoder_layers=config.get('num_decoder_layers', 6),
            num_heads=config.get('num_heads', 8),
            d_ff=config.get('d_ff', 2048),
            forecast_horizon=config.get('forecast_horizon', 24),
            max_seq_len=config.get('max_seq_len', 5000),
            dropout=config.get('dropout', 0.1),
            use_learnable_pe=config.get('use_learnable_pe', False)
        )

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """Forward pass."""
        return self.temporal(src, tgt, src_mask, tgt_mask)

    def forecast(self, x, horizon=None):
        """Generate forecast."""
        return self.temporal.forecast(x, horizon)

    def _save_pretrained(self, save_directory: str):
        """
        Save model to directory.

        Args:
            save_directory: Directory to save to
        """
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_hf_model(
    input_dim: int = 1,
    d_model: int = 512,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    forecast_horizon: int = 24,
    max_seq_len: int = 5000,
    dropout: float = 0.1,
    use_learnable_pe: bool = False,
    use_transformers: bool = True
):
    """
    Create a HuggingFace-compatible Temporal model.

    Args:
        input_dim: Number of input features
        d_model: Model dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        forecast_horizon: Number of steps to forecast
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_learnable_pe: Use learnable positional encoding
        use_transformers: Use transformers PreTrainedModel (True) or PyTorchModelHubMixin (False)

    Returns:
        HuggingFace-compatible model
    """
    if use_transformers:
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers huggingface-hub"
            )

        config = TemporalConfig(
            input_dim=input_dim,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            forecast_horizon=forecast_horizon,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_learnable_pe=use_learnable_pe
        )

        return TemporalForForecasting(config)
    else:
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface-hub library not installed. "
                "Install with: pip install huggingface-hub"
            )

        config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'forecast_horizon': forecast_horizon,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
            'use_learnable_pe': use_learnable_pe
        }

        return TemporalHubMixin(config)
