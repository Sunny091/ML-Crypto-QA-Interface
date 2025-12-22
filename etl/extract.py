"""
Extract Module - 資料提取

Handles the Extract phase of the ETL pipeline.
Retrieves data from various sources (API, mock, file).
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Literal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

Symbol = Literal["BTC", "ETH", "SOL"]


def extract_price_data(
    symbol: Symbol,
    as_of: str | None = None,
    lookback_days: int = 60,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Extract price data from the appropriate source.

    This is the main entry point for the Extract phase.

    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, SOL)
        as_of: Reference date in YYYY-MM-DD format or "now"
        lookback_days: Number of historical days to include
        source: Data source ("api", "mock", "file", or "auto")

    Returns:
        DataFrame with OHLCV columns: date, open, high, low, close, volume
    """
    from data.scrapers.mock_generator import get_mock_price_data

    # Parse as_of date
    if as_of is None or as_of == "now":
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(as_of, "%Y-%m-%d")

    print(f"[Extract] Getting {symbol} data up to {end_date.strftime('%Y-%m-%d')}")

    # For now, use mock data (can be extended to support API/file sources)
    if source in ("mock", "auto"):
        df = get_mock_price_data(
            symbol=symbol,
            as_of=end_date.strftime("%Y-%m-%d"),
            lookback_days=lookback_days,
        )
        print(f"[Extract] Retrieved {len(df)} rows from mock data")
        return df

    # TODO: Implement API source
    # TODO: Implement file source

    raise ValueError(f"Unknown source: {source}")


def extract_historical_data(
    symbol: Symbol,
    start_date: str,
    end_date: str,
    save_to_file: bool = False,
) -> pd.DataFrame:
    """
    Extract historical data for a date range.

    Args:
        symbol: Cryptocurrency symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_to_file: Whether to save to raw data directory

    Returns:
        DataFrame with OHLCV data
    """
    from data.scrapers.mock_generator import generate_mock_price_data

    df = generate_mock_price_data(symbol, start_date, end_date)

    if save_to_file:
        from configs.config import env
        output_path = env.data.raw_data_dir / f"{symbol.lower()}_{start_date}_{end_date}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Extract] Saved to {output_path}")

    return df
