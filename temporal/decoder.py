"""
Decoder layers for Temporal model with residual connections and layer normalization.

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
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feed-forward network.
    Includes residual connections and layer normalization.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward layer
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output tensor (batch_size, src_seq_len, d_model)
            src_mask: Optional source attention mask
            tgt_mask: Optional target attention mask (for causal masking)

        Returns:
            Output tensor (batch_size, tgt_seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-attention with encoder output
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Decoder(nn.Module):
    """
    Stack of decoder layers.

    Args:
        num_layers: Number of decoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward layer
        dropout: Dropout probability
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output tensor (batch_size, src_seq_len, d_model)
            src_mask: Optional source attention mask
            tgt_mask: Optional target attention mask

        Returns:
            Output tensor (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
