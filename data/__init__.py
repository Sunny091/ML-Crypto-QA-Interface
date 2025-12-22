"""
Data Module - DataOps
資料收集、爬蟲和存儲
"""
from data.scrapers.coincap_client import get_current_price, get_price_history
from data.scrapers.mock_generator import generate_mock_price_data, get_mock_price_data

__all__ = [
    "get_current_price",
    "get_price_history",
    "generate_mock_price_data",
    "get_mock_price_data",
]
