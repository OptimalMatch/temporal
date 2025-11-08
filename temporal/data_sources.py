"""
Data source utilities for fetching financial time series data.

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

Provides functions to fetch and prepare stock, crypto, and other financial data
for time series forecasting with the Temporal model.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
from datetime import datetime, timedelta


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch stock price data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        period: Time period to fetch. Options: '1d', '5d', '1mo', '3mo', '6mo',
                '1y', '2y', '5y', '10y', 'ytd', 'max'
        interval: Data interval. Options: '1m', '2m', '5m', '15m', '30m', '60m',
                  '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        start_date: Start date (format: 'YYYY-MM-DD'). If provided, period is ignored
        end_date: End date (format: 'YYYY-MM-DD'). If not provided, uses current date

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume

    Example:
        >>> df = fetch_stock_data('AAPL', period='1y', interval='1d')
        >>> print(df.head())
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for fetching stock data. "
            "Install with: pip install yfinance"
        )

    # Download data
    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    elif start_date:
        data = yf.download(ticker, start=start_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)

    # Reset index to make Date a column
    data = data.reset_index()

    # Standardize column names based on what's available
    # yfinance may return different columns depending on the asset type
    if len(data.columns) == 7:
        # Standard format with Adj Close
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    elif len(data.columns) == 6:
        # Some assets (like crypto) don't have Adj Close
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        # Add Adj Close as a copy of Close for consistency
        data['Adj Close'] = data['Close']
    else:
        # Fallback: keep original column names
        pass

    return data


def fetch_crypto_data(
    symbol: str = "BTC-USD",
    period: str = "2y",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch cryptocurrency price data from Yahoo Finance.

    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD', 'DOGE-USD')
        period: Time period to fetch
        interval: Data interval
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume

    Example:
        >>> df = fetch_crypto_data('BTC-USD', period='1y')
        >>> print(df.head())
    """
    # Use the same function as stocks
    return fetch_stock_data(symbol, period, interval, start_date, end_date)


def fetch_multiple_stocks(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
    column: str = "Close"
) -> pd.DataFrame:
    """
    Fetch multiple stock prices and combine into multivariate dataset.

    Args:
        tickers: List of stock ticker symbols
        period: Time period to fetch
        interval: Data interval
        column: Which column to use ('Open', 'High', 'Low', 'Close', 'Volume')

    Returns:
        DataFrame with Date index and columns for each ticker

    Example:
        >>> df = fetch_multiple_stocks(['AAPL', 'GOOGL', 'MSFT'], period='1y')
        >>> print(df.head())
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required. Install with: pip install yfinance"
        )

    # Download all tickers at once
    data = yf.download(tickers, period=period, interval=interval)

    # Extract the desired column
    if len(tickers) == 1:
        result = pd.DataFrame({
            tickers[0]: data[column]
        })
    else:
        result = data[column]

    # Reset index
    result = result.reset_index()

    return result


def prepare_for_temporal(
    df: pd.DataFrame,
    feature_columns: Union[str, List[str]],
    date_column: str = 'Date',
    dropna: bool = True
) -> np.ndarray:
    """
    Prepare DataFrame for Temporal model training.

    Args:
        df: Input DataFrame
        feature_columns: Column name(s) to use as features
        date_column: Name of the date column (will be dropped)
        dropna: Whether to drop NaN values

    Returns:
        NumPy array of shape (num_samples, num_features)

    Example:
        >>> df = fetch_stock_data('AAPL', period='1y')
        >>> data = prepare_for_temporal(df, feature_columns='Close')
        >>> print(data.shape)  # (num_days, 1)
    """
    # Handle single column or list of columns
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]

    # Extract features
    data = df[feature_columns].values

    # Drop NaN if requested
    if dropna:
        data = data[~np.isnan(data).any(axis=1)]

    # Ensure 2D shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data


def add_technical_indicators(
    df: pd.DataFrame,
    close_col: str = 'Close'
) -> pd.DataFrame:
    """
    Add common technical indicators to price data.

    Adds:
    - Simple Moving Averages (SMA): 7-day, 21-day, 50-day
    - Exponential Moving Averages (EMA): 12-day, 26-day
    - Relative Strength Index (RSI): 14-day
    - Bollinger Bands
    - Daily Returns
    - Volume Change

    Args:
        df: DataFrame with stock/crypto data
        close_col: Name of the closing price column

    Returns:
        DataFrame with additional technical indicator columns

    Example:
        >>> df = fetch_stock_data('AAPL', period='1y')
        >>> df_with_indicators = add_technical_indicators(df)
        >>> print(df_with_indicators.columns)
    """
    df = df.copy()

    # Simple Moving Averages
    df['SMA_7'] = df[close_col].rolling(window=7).mean()
    df['SMA_21'] = df[close_col].rolling(window=21).mean()
    df['SMA_50'] = df[close_col].rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df[close_col].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df[close_col].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (14-day)
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df[close_col].rolling(window=20).mean()
    bb_std = df[close_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Daily Returns
    df['Returns'] = df[close_col].pct_change()

    # Volume Change
    if 'Volume' in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change()

    return df


def split_train_val_test(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train, validation, and test sets.

    Args:
        data: Input data array
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_data, val_data, test_data)

    Example:
        >>> data = prepare_for_temporal(df, 'Close')
        >>> train, val, test = split_train_val_test(data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def create_sequences(
    data: np.ndarray,
    lookback: int,
    forecast_horizon: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for time series forecasting.

    Args:
        data: Input time series data (num_samples, num_features)
        lookback: Number of historical steps to use as input
        forecast_horizon: Number of future steps to predict
        stride: Step size for sliding window

    Returns:
        Tuple of (X, y) where:
        - X: Input sequences (num_sequences, lookback, num_features)
        - y: Target sequences (num_sequences, forecast_horizon, num_features)

    Example:
        >>> X, y = create_sequences(data, lookback=96, forecast_horizon=24)
        >>> print(X.shape, y.shape)
    """
    X, y = [], []

    for i in range(0, len(data) - lookback - forecast_horizon + 1, stride):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + forecast_horizon])

    return np.array(X), np.array(y)


def get_sample_data(dataset: str = 'stocks') -> pd.DataFrame:
    """
    Get sample data for quick testing.

    Args:
        dataset: Type of sample data ('stocks', 'crypto', 'multi_stocks')

    Returns:
        Sample DataFrame

    Example:
        >>> df = get_sample_data('stocks')
        >>> print(df.head())
    """
    if dataset == 'stocks':
        return fetch_stock_data('AAPL', period='6mo', interval='1d')
    elif dataset == 'crypto':
        return fetch_crypto_data('BTC-USD', period='6mo', interval='1d')
    elif dataset == 'multi_stocks':
        return fetch_multiple_stocks(['AAPL', 'GOOGL', 'MSFT'], period='6mo')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def fetch_kaggle_bitcoin_data(
    dataset_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch Bitcoin historical data from Kaggle.

    Uses the "mczielinski/bitcoin-historical-data" dataset which provides
    minute-level Bitcoin data from multiple exchanges.

    Args:
        dataset_path: Path to downloaded dataset. If None, downloads automatically.

    Returns:
        DataFrame with Bitcoin price data

    Example:
        >>> df = fetch_kaggle_bitcoin_data()
        >>> print(df.head())

    Note:
        Requires kagglehub: pip install kagglehub
        You may need to configure Kaggle API credentials.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required for Kaggle datasets. "
            "Install with: pip install kagglehub"
        )

    # Download dataset if path not provided
    if dataset_path is None:
        print("Downloading Bitcoin dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
        print(f"Dataset downloaded to: {dataset_path}")

    # Load the bitstamp data (most complete)
    import os
    bitstamp_file = os.path.join(dataset_path, "bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")

    if not os.path.exists(bitstamp_file):
        # Try to find any CSV file in the directory
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if csv_files:
            bitstamp_file = os.path.join(dataset_path, csv_files[0])
        else:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    # Load data
    df = pd.read_csv(bitstamp_file)

    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.rename(columns={'Timestamp': 'Date'})

    # Rename columns to match standard format
    column_mapping = {
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume_(BTC)': 'Volume_BTC',
        'Volume_(Currency)': 'Volume',
        'Weighted_Price': 'Weighted_Price'
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    # Remove rows with NaN in critical columns
    df = df.dropna(subset=['Close'])

    return df


def resample_bitcoin_data(
    df: pd.DataFrame,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Resample minute-level Bitcoin data to different intervals.

    Args:
        df: DataFrame from fetch_kaggle_bitcoin_data()
        interval: Resampling interval ('1h', '1d', '1w', '1mo')

    Returns:
        Resampled DataFrame

    Example:
        >>> df = fetch_kaggle_bitcoin_data()
        >>> df_daily = resample_bitcoin_data(df, interval='1d')
    """
    df = df.copy()
    df = df.set_index('Date')

    # Resample OHLCV data
    resampled = df.resample(interval).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Remove NaN rows
    resampled = resampled.dropna()

    # Reset index
    resampled = resampled.reset_index()

    return resampled


def normalize_data(
    data: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, object]:
    """
    Normalize data for training.

    Args:
        data: Input data
        method: Normalization method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (normalized_data, scaler)

    Example:
        >>> data_norm, scaler = normalize_data(data, method='standard')
        >>> # Later: data_original = scaler.inverse_transform(data_norm)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    data_normalized = scaler.fit_transform(data)

    return data_normalized, scaler
