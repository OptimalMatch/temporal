"""
Training utilities for Temporal model.

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
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, List
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting.

    Args:
        data: Time series data (num_samples, seq_len, num_features)
        lookback: Number of historical time steps to use
        forecast_horizon: Number of future time steps to predict
        stride: Stride for sliding window
    """

    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 96,
        forecast_horizon: int = 24,
        stride: int = 1
    ):
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.stride = stride

        # Calculate number of samples
        self.num_samples = (len(data) - lookback - forecast_horizon) // stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        src_end_idx = start_idx + self.lookback
        tgt_end_idx = src_end_idx + self.forecast_horizon

        # Source: historical data
        src = self.data[start_idx:src_end_idx]

        # Target: future data (shifted by 1 for teacher forcing)
        # Decoder input: last point of source + first (horizon-1) points of target
        decoder_input = self.data[src_end_idx - 1:tgt_end_idx - 1]

        # Target output: actual future values
        target_output = self.data[src_end_idx:tgt_end_idx]

        return src, decoder_input, target_output


class TemporalTrainer:
    """
    Trainer for Temporal model.

    Args:
        model: Temporal model instance
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on
        grad_clip: Gradient clipping value (None to disable)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: Optional[float] = 1.0,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp and torch.cuda.is_available()

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler() if self.use_amp else None

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        num_skipped = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for src, decoder_input, target_output in progress_bar:
            src = src.to(self.device)
            decoder_input = decoder_input.to(self.device)
            target_output = target_output.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_amp:
                with autocast():
                    output = self.model(src, decoder_input)
                    loss = self.criterion(output, target_output)

                # Check for NaN loss - skip this batch if NaN to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    num_skipped += 1
                    progress_bar.set_postfix({"loss": float('nan'), "skipped": num_skipped})
                    continue

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(src, decoder_input)
                loss = self.criterion(output, target_output)

                # Check for NaN loss - skip this batch if NaN to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    num_skipped += 1
                    progress_bar.set_postfix({"loss": float('nan'), "skipped": num_skipped})
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": loss.item()})

        if num_skipped > 0:
            print(f"⚠️  Skipped {num_skipped} batches with NaN/inf loss out of {num_batches + num_skipped} total")

        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for src, decoder_input, target_output in dataloader:
                src = src.to(self.device)
                decoder_input = decoder_input.to(self.device)
                target_output = target_output.to(self.device)

                # Forward pass
                output = self.model(src, decoder_input)

                # Compute loss
                loss = self.criterion(output, target_output)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: Optional[int] = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            save_path: Path to save the best model

        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.6f}")

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Val Loss: {val_loss:.6f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    if save_path is not None:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Model saved to {save_path}")
                else:
                    patience_counter += 1

                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Generate predictions for a dataset.

        Args:
            dataloader: Data loader

        Returns:
            Array of predictions
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for src, _, _ in dataloader:
                src = src.to(self.device)

                # Generate forecast (autoregressive)
                output = self.model.forecast(src)
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)
