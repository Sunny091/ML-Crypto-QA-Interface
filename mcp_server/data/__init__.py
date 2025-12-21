"""
Data module for price data retrieval.
Phase 2: Mock data only.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Literal

Symbol = Literal["BTC", "ETH", "SOL"]

# Seed for reproducibility
RANDOM_SEED = 42


def generate_mock_price_data(
    symbol: Symbol,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Generate deterministic mock OHLCV price data for a given symbol.

    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, SOL)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(RANDOM_SEED + hash(symbol) % 1000)

    # Base prices for each symbol
    base_prices = {
        "BTC": 30000.0,
        "ETH": 2000.0,
        "SOL": 50.0,
    }

    base_price = base_prices[symbol]

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    # Generate price series with random walk
    returns = np.random.normal(0.0005, 0.03, n_days)  # ~0.05% daily drift, 3% volatility
    cumulative_returns = np.cumprod(1 + returns)
    close_prices = base_price * cumulative_returns

    # Generate OHLC from close
    daily_volatility = np.abs(np.random.normal(0, 0.02, n_days))

    open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n_days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + daily_volatility)
    low_prices = np.minimum(open_prices, close_prices) * (1 - daily_volatility)

    # Generate volume
    base_volume = {"BTC": 1e9, "ETH": 5e8, "SOL": 1e8}[symbol]
    volumes = base_volume * np.abs(np.random.normal(1, 0.3, n_days))

    df = pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })

    return df


def get_price_data(
    symbol: Symbol,
    as_of: str | None = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Get price data for a symbol up to a given date.

    Args:
        symbol: Cryptocurrency symbol
        as_of: Reference date in YYYY-MM-DD format (default: today)
        lookback_days: Number of historical days to include

    Returns:
        DataFrame with OHLCV data
    """
    # Parse as_of date
    if as_of is None or as_of == "now":
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(as_of, "%Y-%m-%d")

    start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer

    # Get full mock data
    df = generate_mock_price_data(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    # Filter to requested range
    df = df[df["date"] <= pd.Timestamp(end_date)]
    df = df.tail(lookback_days)

    return df.reset_index(drop=True)
