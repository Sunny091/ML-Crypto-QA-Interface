"""
CoinGecko API Client Module - 資料爬蟲

Provides functions for fetching real-time and historical cryptocurrency data.
API: https://api.coingecko.com/api/v3/
No API key required for basic usage.

This module handles:
- Extract phase of ETL pipeline
- Real-time price data collection
- Historical price data collection
"""

import sys
import ssl
from datetime import datetime, timedelta
from typing import Literal
import urllib.request
import urllib.error
import json

# SSL context for macOS certificate issues
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Symbol to CoinGecko ID mapping
SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

Symbol = Literal["BTC", "ETH", "SOL"]


class CoinGeckoClient:
    """Synchronous client for CoinGecko API - 資料爬蟲"""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make HTTP GET request to CoinGecko API"""
        url = f"{COINGECKO_BASE_URL}{endpoint}"

        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{query}"

        print(f"[CoinGecko] GET {url}", file=sys.stderr)

        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            req.add_header('User-Agent', 'Mozilla/5.0')

            with urllib.request.urlopen(req, timeout=self.timeout, context=SSL_CONTEXT) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data
        except urllib.error.HTTPError as e:
            print(f"[CoinGecko] HTTP Error: {e.code}", file=sys.stderr)
            raise
        except urllib.error.URLError as e:
            print(f"[CoinGecko] URL Error: {e.reason}", file=sys.stderr)
            raise

    def get_asset(self, symbol: str) -> dict:
        """
        Get current asset data including price, volume, market cap.

        Args:
            symbol: Crypto symbol (BTC, ETH, SOL)

        Returns:
            Asset data dict with priceUsd, changePercent24Hr, volumeUsd24Hr, marketCapUsd
        """
        asset_id = SYMBOL_MAP.get(symbol.upper())
        if not asset_id:
            raise ValueError(f"Unknown symbol: {symbol}")

        params = {
            "ids": asset_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
            "include_market_cap": "true",
        }
        response = self._request("/simple/price", params)

        data = response.get(asset_id, {})
        return {
            "priceUsd": data.get("usd", 0),
            "changePercent24Hr": data.get("usd_24h_change", 0),
            "volumeUsd24Hr": data.get("usd_24h_vol", 0),
            "marketCapUsd": data.get("usd_market_cap", 0),
        }

    def get_history(
        self,
        symbol: str,
        interval: Literal["m1", "m5", "m15", "m30", "h1", "h2", "h6", "h12", "d1"] = "d1",
        days: int = None,
        start_date: str = None,
        end_date: str = None,
    ) -> list[dict]:
        """
        Get historical price data.

        Args:
            symbol: Crypto symbol (BTC, ETH, SOL)
            interval: Time interval (ignored for CoinGecko, auto-determined by days)
            days: Number of days to look back (1-365)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of price points with priceUsd and time
        """
        asset_id = SYMBOL_MAP.get(symbol.upper())
        if not asset_id:
            raise ValueError(f"Unknown symbol: {symbol}")

        # Calculate days from date range if provided
        if start_date and end_date:
            start_time = datetime.strptime(start_date, "%Y-%m-%d")
            end_time = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_time - start_time).days + 1
        else:
            days = days or 30

        params = {
            "vs_currency": "usd",
            "days": str(days),
        }

        response = self._request(f"/coins/{asset_id}/market_chart", params)
        prices = response.get("prices", [])

        # Convert to standard format
        return [{"time": p[0], "priceUsd": str(p[1])} for p in prices]


# Singleton instance
_client: CoinGeckoClient | None = None


def get_coingecko_client() -> CoinGeckoClient:
    """Get or create singleton CoinGecko client"""
    global _client
    if _client is None:
        _client = CoinGeckoClient()
    return _client


# Alias for backward compatibility
def get_coincap_client() -> CoinGeckoClient:
    """Alias for get_coingecko_client (backward compatibility)"""
    return get_coingecko_client()


def _get_mock_price(symbol: str) -> dict:
    """Return mock price data when API is unavailable"""
    mock_prices = {
        "BTC": {"price": 97250.00, "change": 2.15, "volume": 28.5e9, "market_cap": 1920e9},
        "ETH": {"price": 3420.50, "change": 1.85, "volume": 15.2e9, "market_cap": 412e9},
        "SOL": {"price": 192.30, "change": -0.75, "volume": 3.8e9, "market_cap": 89e9},
    }
    data = mock_prices.get(symbol.upper(), mock_prices["BTC"])
    return {
        "symbol": symbol.upper(),
        "price_usd": data["price"],
        "change_24h_percent": data["change"],
        "volume_24h_usd": data["volume"],
        "market_cap_usd": data["market_cap"],
        "timestamp": datetime.now().isoformat(),
        "source": "mock",
    }


def get_current_price(symbol: str) -> dict:
    """
    Get current price data for a symbol.

    Args:
        symbol: BTC, ETH, or SOL

    Returns:
        Dict with price_usd, change_24h_percent, volume_24h_usd, market_cap_usd, timestamp
    """
    try:
        client = get_coincap_client()
        data = client.get_asset(symbol)

        return {
            "symbol": symbol.upper(),
            "price_usd": float(data.get("priceUsd", 0)),
            "change_24h_percent": float(data.get("changePercent24Hr", 0)),
            "volume_24h_usd": float(data.get("volumeUsd24Hr", 0)),
            "market_cap_usd": float(data.get("marketCapUsd", 0)),
            "timestamp": datetime.now().isoformat(),
            "source": "coingecko",
        }
    except Exception as e:
        print(f"[CoinGecko] API failed: {e}, using mock data", file=sys.stderr)
        return _get_mock_price(symbol)


def get_price_history(
    symbol: str,
    interval: str = "d1",
    days: int = None,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Get historical price data for a symbol.

    Args:
        symbol: BTC, ETH, or SOL
        interval: h1, h2, h6, h12, d1
        days: Number of days (1-365)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict with symbol, interval, data list, start_date, end_date
    """
    from data.scrapers.mock_generator import get_mock_history

    # Determine date range
    if start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days or 30)

    try:
        client = get_coincap_client()
        raw_data = client.get_history(
            symbol, interval,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d")
        )

        # Format data points
        data_points = []
        for point in raw_data:
            timestamp = datetime.fromtimestamp(point["time"] / 1000)
            data_points.append({
                "timestamp": timestamp.isoformat(),
                "price_usd": float(point["priceUsd"]),
            })

        source = "coingecko"
    except Exception as e:
        print(f"[CoinGecko] History API failed: {e}, using mock data", file=sys.stderr)
        data_points = get_mock_history(symbol, start_dt, end_dt)
        source = "mock"

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "data": data_points,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "source": source,
    }
