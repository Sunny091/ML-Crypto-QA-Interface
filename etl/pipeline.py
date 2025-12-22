"""
ETL Pipeline - 完整的資料處理流程

Orchestrates the full Extract-Transform-Load pipeline.
Automates data collection, cleaning, and feature engineering.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.extract import extract_price_data, extract_historical_data
from etl.transform import build_features, clean_data, prepare_training_data

Symbol = Literal["BTC", "ETH", "SOL"]


class ETLPipeline:
    """
    Complete ETL Pipeline for cryptocurrency price prediction.

    Stages:
    1. Extract: Get raw price data from sources
    2. Transform: Clean data and engineer features
    3. Load: Save processed data for model training
    """

    def __init__(self, symbols: list[Symbol] = None):
        """
        Initialize the ETL pipeline.

        Args:
            symbols: List of cryptocurrency symbols to process
        """
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        self.raw_data: dict[str, pd.DataFrame] = {}
        self.processed_data: dict[str, pd.DataFrame] = {}

    def extract(
        self,
        start_date: str = "2022-01-01",
        end_date: str = None,
    ) -> "ETLPipeline":
        """
        Extract phase: Collect raw data from sources.

        Args:
            start_date: Start date for historical data
            end_date: End date (default: today)

        Returns:
            Self for method chaining
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"ETL Pipeline - Extract Phase")
        print(f"{'='*60}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Symbols: {self.symbols}")

        for symbol in self.symbols:
            print(f"\n[Extract] Processing {symbol}...")
            df = extract_historical_data(symbol, start_date, end_date)
            self.raw_data[symbol] = df
            print(f"[Extract] {symbol}: {len(df)} rows extracted")

        return self

    def transform(self) -> "ETLPipeline":
        """
        Transform phase: Clean data and engineer features.

        Returns:
            Self for method chaining
        """
        print(f"\n{'='*60}")
        print(f"ETL Pipeline - Transform Phase")
        print(f"{'='*60}")

        for symbol, df in self.raw_data.items():
            print(f"\n[Transform] Processing {symbol}...")

            # Clean data
            df_clean = clean_data(df)
            print(f"[Transform] After cleaning: {len(df_clean)} rows")

            # Build features
            df_features = build_features(df_clean)
            print(f"[Transform] Features added: {list(df_features.columns)}")

            self.processed_data[symbol] = df_features

        return self

    def load(self, output_dir: Path = None) -> "ETLPipeline":
        """
        Load phase: Save processed data to files.

        Args:
            output_dir: Directory to save processed data

        Returns:
            Self for method chaining
        """
        from configs.config import env

        if output_dir is None:
            output_dir = env.data.processed_data_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"ETL Pipeline - Load Phase")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")

        for symbol, df in self.processed_data.items():
            output_path = output_dir / f"{symbol.lower()}_processed.csv"
            df.to_csv(output_path, index=False)
            print(f"[Load] Saved {symbol} to {output_path}")

        return self

    def run(
        self,
        start_date: str = "2022-01-01",
        end_date: str = None,
        save: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Run the complete ETL pipeline.

        Args:
            start_date: Start date for historical data
            end_date: End date (default: today)
            save: Whether to save processed data to files

        Returns:
            Dictionary of symbol -> processed DataFrame
        """
        self.extract(start_date, end_date)
        self.transform()

        if save:
            self.load()

        print(f"\n{'='*60}")
        print(f"ETL Pipeline Complete!")
        print(f"{'='*60}")

        return self.processed_data

    def get_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get combined training data from all symbols.

        Returns:
            Tuple of (X features, y labels)
        """
        all_X = []
        all_y = []

        for symbol, df in self.processed_data.items():
            X, y = prepare_training_data(df)
            all_X.append(X)
            all_y.append(y)

        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)

        return X_combined, y_combined


def run_etl_pipeline():
    """CLI entry point for running the ETL pipeline."""
    pipeline = ETLPipeline()
    pipeline.run()


if __name__ == "__main__":
    run_etl_pipeline()
