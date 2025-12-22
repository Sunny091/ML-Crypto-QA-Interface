"""
Transform Module - 特徵工程與資料清理

Handles the Transform phase of the ETL pipeline.
Calculates technical indicators and prepares features for ML models.
"""

import pandas as pd
import numpy as np
from typing import Literal

# Feature names used in this module
PRICE_ONLY_FEATURES = ["return_1d", "rsi_14", "volatility_7d"]
PRICE_TEXT_FEATURES = ["return_1d", "rsi_14", "volatility_7d", "sentiment_score", "news_present"]


def calculate_return_1d(df: pd.DataFrame) -> pd.Series:
    """
    Calculate 1-day return (percentage change from previous close).

    Formula: (close_t - close_{t-1}) / close_{t-1}
    """
    return df["close"].pct_change(1)


def calculate_rsi_14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) with 14-day period.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss over period
    """
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_volatility_7d(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """
    Calculate 7-day rolling volatility (standard deviation of returns).

    Formula: std(return_1d) over 7-day window
    """
    returns = df["close"].pct_change(1)
    volatility = returns.rolling(window=period, min_periods=period).std()

    return volatility


def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return df["close"].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return df["close"].ewm(span=period, adjust=False).mean()


def calculate_macd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all price-only features from OHLCV data.

    This is the main Transform function that applies feature engineering.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume

    Returns:
        DataFrame with original columns plus feature columns
    """
    df = df.copy()

    # Calculate core features
    df["return_1d"] = calculate_return_1d(df)
    df["rsi_14"] = calculate_rsi_14(df)
    df["volatility_7d"] = calculate_volatility_7d(df)

    # Additional technical indicators (optional)
    df["sma_20"] = calculate_sma(df, 20)
    df["sma_50"] = calculate_sma(df, 50)
    df["ema_12"] = calculate_ema(df, 12)
    df["ema_26"] = calculate_ema(df, 26)

    return df


def get_latest_features(df: pd.DataFrame) -> dict[str, float]:
    """
    Get the latest feature values from a DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary of feature name -> value
    """
    df_with_features = build_features(df)

    # Get the last row
    last_row = df_with_features.iloc[-1]

    return {
        "return_1d": float(last_row["return_1d"]) if pd.notna(last_row["return_1d"]) else 0.0,
        "rsi_14": float(last_row["rsi_14"]) if pd.notna(last_row["rsi_14"]) else 50.0,
        "volatility_7d": float(last_row["volatility_7d"]) if pd.notna(last_row["volatility_7d"]) else 0.02,
    }


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for model training (price-only).

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Tuple of (X features DataFrame, y labels Series)
    """
    df = build_features(df)

    # Create target: 1 if next day close > today close, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop rows with NaN (first few rows due to rolling windows, last row due to target)
    df = df.dropna()

    X = df[PRICE_ONLY_FEATURES]
    y = df["target"]

    return X, y


def get_price_text_features(
    df: pd.DataFrame,
    news_text: str | None,
) -> dict[str, float]:
    """
    Get combined price and text features for prediction.

    Args:
        df: DataFrame with OHLCV data
        news_text: Optional news text for sentiment analysis

    Returns:
        Dictionary with all price+text features
    """
    from models.sentiment import get_news_features

    # Get price features
    price_features = get_latest_features(df)

    # Get news features
    news_features = get_news_features(news_text)

    # Combine
    return {
        **price_features,
        **news_features,
    }


def prepare_training_data_with_text(
    df: pd.DataFrame,
    sentiment_scores: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for price+text model training.

    Args:
        df: DataFrame with OHLCV data
        sentiment_scores: Optional list of sentiment scores per row

    Returns:
        Tuple of (X features DataFrame, y labels Series)
    """
    df = build_features(df)

    # Create target: 1 if next day close > today close, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Add text features
    if sentiment_scores is not None and len(sentiment_scores) == len(df):
        df["sentiment_score"] = sentiment_scores
        df["news_present"] = [1.0 if s != 0 else 0.0 for s in sentiment_scores]
    else:
        # Generate mock sentiment data for training
        np.random.seed(42)
        base_sentiment = np.random.uniform(-0.5, 0.5, len(df))
        df["sentiment_score"] = base_sentiment
        df["news_present"] = np.random.choice([0.0, 1.0], len(df), p=[0.3, 0.7])
        df.loc[df["news_present"] == 0, "sentiment_score"] = 0.0

    # Drop rows with NaN
    df = df.dropna()

    X = df[PRICE_TEXT_FEATURES]
    y = df["target"]

    return X, y


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by handling missing values and outliers.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Remove rows with missing values
    df = df.dropna()

    # Remove extreme outliers (beyond 5 standard deviations)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 5 * std) & (df[col] <= mean + 5 * std)]

    return df.reset_index(drop=True)
