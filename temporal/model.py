"""
Temporal: A transformer-based model for time series forecasting.

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

The model uses self-attention mechanisms to capture temporal dependencies
and patterns in time series data for accurate forecasting.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .encoder import Encoder
from .decoder import Decoder
from .position_encoding import TemporalPositionEncoding


class Temporal(nn.Module):
    """
    Temporal: Transformer-based time series forecasting model.

    Architecture:
    - Input embedding layer to project time series to model dimension
    - Positional encoding to capture temporal order
    - Encoder stack with self-attention and feed-forward layers
    - Decoder stack with self-attention, cross-attention, and feed-forward layers
    - Output projection layer to map to forecasting horizon

    Args:
        input_dim: Number of input features (e.g., 1 for univariate, >1 for multivariate)
        d_model: Dimension of the model (hidden size)
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        forecast_horizon: Number of time steps to forecast
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_learnable_pe: Use learnable positional encoding instead of sinusoidal
    """

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
        use_learnable_pe: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Input embedding: project input features to model dimension
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        if use_learnable_pe:
            from .position_encoding import LearnablePositionEncoding
            self.position_encoding = LearnablePositionEncoding(d_model, max_seq_len, dropout)
        else:
            self.position_encoding = TemporalPositionEncoding(d_model, max_seq_len, dropout)

        # Encoder
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # Decoder
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # Output projection: map decoder output to forecast dimension
        self.output_projection = nn.Linear(d_model, input_dim)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """
        Generate causal mask for decoder self-attention.
        Prevents attending to future positions.

        Args:
            size: Sequence length

        Returns:
            Causal mask tensor of shape (size, size)
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Temporal model.

        Args:
            src: Source time series (batch_size, src_seq_len, input_dim)
            tgt: Target time series for teacher forcing (batch_size, tgt_seq_len, input_dim)
                 If None, uses autoregressive generation
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Forecasted values (batch_size, forecast_horizon, input_dim)
        """
        # Embed and add positional encoding to source
        src_embedded = self.input_embedding(src)
        src_encoded = self.position_encoding(src_embedded)

        # Encode source sequence
        encoder_output = self.encoder(src_encoded, src_mask)

        # If target is provided, use teacher forcing
        if tgt is not None:
            return self._forward_with_teacher_forcing(encoder_output, tgt, src_mask, tgt_mask)
        else:
            # Autoregressive generation for inference
            return self._forward_autoregressive(encoder_output, src, src_mask)

    def _forward_with_teacher_forcing(
        self,
        encoder_output: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with teacher forcing (training mode)."""
        # Embed and add positional encoding to target
        tgt_embedded = self.input_embedding(tgt)
        tgt_encoded = self.position_encoding(tgt_embedded)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.generate_causal_mask(tgt_seq_len).to(tgt.device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

        # Decode
        decoder_output = self.decoder(tgt_encoded, encoder_output, src_mask, tgt_mask)

        # Project to output dimension
        output = self.output_projection(decoder_output)

        return output

    def _forward_autoregressive(
        self,
        encoder_output: torch.Tensor,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with autoregressive generation (inference mode)."""
        batch_size = src.size(0)
        device = src.device

        # Start with the last value of the source sequence
        decoder_input = src[:, -1:, :]  # (batch_size, 1, input_dim)

        outputs = []

        for _ in range(self.forecast_horizon):
            # Embed and encode decoder input
            tgt_embedded = self.input_embedding(decoder_input)
            tgt_encoded = self.position_encoding(tgt_embedded)

            # Generate causal mask
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = self.generate_causal_mask(tgt_seq_len).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # Decode
            decoder_output = self.decoder(tgt_encoded, encoder_output, src_mask, tgt_mask)

            # Project to output dimension
            output = self.output_projection(decoder_output[:, -1:, :])  # Get last time step

            outputs.append(output)

            # Append to decoder input for next iteration
            decoder_input = torch.cat([decoder_input, output], dim=1)

        # Concatenate all predictions
        forecast = torch.cat(outputs, dim=1)

        return forecast

    def forecast(self, x: torch.Tensor, horizon: Optional[int] = None) -> torch.Tensor:
        """
        Generate forecasts for the given time series.

        Args:
            x: Input time series (batch_size, seq_len, input_dim)
            horizon: Forecast horizon (uses model default if None)

        Returns:
            Forecasted values (batch_size, horizon, input_dim)
        """
        self.eval()
        with torch.no_grad():
            if horizon is not None and horizon != self.forecast_horizon:
                # Temporarily change forecast horizon
                original_horizon = self.forecast_horizon
                self.forecast_horizon = horizon
                output = self.forward(x)
                self.forecast_horizon = original_horizon
            else:
                output = self.forward(x)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> Tuple[list, list]:
        """
        Extract attention weights from encoder and decoder for visualization.

        Args:
            x: Input time series (batch_size, seq_len, input_dim)

        Returns:
            Tuple of (encoder_attention_weights, decoder_attention_weights)
        """
        # This is a simplified version - full implementation would require
        # modifying forward pass to return attention weights
        raise NotImplementedError("Attention weight extraction not yet implemented")
