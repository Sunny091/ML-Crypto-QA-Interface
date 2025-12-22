"""Data Scrapers - 資料爬蟲模組"""
from data.scrapers.coincap_client import CoinGeckoClient, get_current_price, get_price_history
from data.scrapers.mock_generator import generate_mock_price_data, get_mock_price_data

# Backward compatibility alias
CoinCapClient = CoinGeckoClient

__all__ = [
    "CoinGeckoClient",
    "CoinCapClient",  # alias for backward compatibility
    "get_current_price",
    "get_price_history",
    "generate_mock_price_data",
    "get_mock_price_data",
]
