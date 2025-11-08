"""
Utility functions for Temporal model.

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
import numpy as np
from typing import Tuple, Optional


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    Create a look-ahead mask to prevent attention to future positions.
    Used in decoder self-attention.

    Args:
        size: Sequence length

    Returns:
        Boolean mask tensor of shape (size, size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask


def create_padding_mask(seq: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """
    Create a padding mask for sequences with padding values.

    Args:
        seq: Input sequence (batch_size, seq_len)
        pad_value: Value used for padding

    Returns:
        Boolean mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_value).unsqueeze(1).unsqueeze(2)
    return mask


def normalize_data(data: np.ndarray, method: str = "standard") -> Tuple[np.ndarray, dict]:
    """
    Normalize time series data.

    Args:
        data: Input data (num_samples, num_features)
        method: Normalization method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        params = {"mean": mean, "std": std, "method": "standard"}

    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {"min": min_val, "max": max_val, "method": "minmax"}

    elif method == "robust":
        median = np.median(data, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        iqr = q75 - q25
        normalized = (data - median) / (iqr + 1e-8)
        params = {"median": median, "iqr": iqr, "method": "robust"}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize data using saved parameters.

    Args:
        data: Normalized data
        params: Dictionary with normalization parameters

    Returns:
        Original scale data
    """
    method = params["method"]

    if method == "standard":
        return data * params["std"] + params["mean"]

    elif method == "minmax":
        return data * (params["max"] - params["min"]) + params["min"]

    elif method == "robust":
        return data * params["iqr"] + params["median"]

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def split_train_val_test(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train, validation, and test sets.

    Args:
        data: Time series data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Calculate common forecasting metrics.

    Args:
        predictions: Predicted values
        targets: Actual values

    Returns:
        Dictionary with metric names and values
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' - whether lower or higher is better
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss

        elif self.mode == "min":
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

        elif self.mode == "max":
            if val_loss > self.best_loss + self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class LearningRateScheduler:
    """
    Custom learning rate scheduler with warmup.

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """Update learning rate."""
        self.step_num += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """Calculate learning rate."""
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
