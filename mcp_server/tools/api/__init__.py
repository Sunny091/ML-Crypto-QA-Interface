"""
CoinCap API Client Module

Provides functions for fetching real-time and historical cryptocurrency data.
API: https://api.coincap.io/v2/
No API key required, no rate limits.
"""

import sys
from datetime import datetime, timedelta
from typing import Literal
import urllib.request
import urllib.error
import json

COINCAP_BASE_URL = "https://api.coincap.io/v2"

# Symbol to CoinCap ID mapping
SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}


class CoinCapClient:
    """Synchronous client for CoinCap API"""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make HTTP GET request to CoinCap API"""
        url = f"{COINCAP_BASE_URL}{endpoint}"

        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{query}"

        print(f"[CoinCap] GET {url}", file=sys.stderr)

        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data
        except urllib.error.HTTPError as e:
            print(f"[CoinCap] HTTP Error: {e.code}", file=sys.stderr)
            raise
        except urllib.error.URLError as e:
            print(f"[CoinCap] URL Error: {e.reason}", file=sys.stderr)
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

        response = self._request(f"/assets/{asset_id}")
        return response.get("data", {})

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
            interval: Time interval (m1, m5, m15, m30, h1, h2, h6, h12, d1)
            days: Number of days to look back (1-365), used if start_date/end_date not provided
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of price points with priceUsd and time
        """
        asset_id = SYMBOL_MAP.get(symbol.upper())
        if not asset_id:
            raise ValueError(f"Unknown symbol: {symbol}")

        # Calculate time range
        if start_date and end_date:
            # Use specified date range
            start_time = datetime.strptime(start_date, "%Y-%m-%d")
            end_time = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # Include end date
        else:
            # Use days parameter (default 30)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days or 30)

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        params = {
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
        }

        response = self._request(f"/assets/{asset_id}/history", params)
        return response.get("data", [])


# Singleton instance
_client: CoinCapClient | None = None


def get_coincap_client() -> CoinCapClient:
    """Get or create singleton CoinCap client"""
    global _client
    if _client is None:
        _client = CoinCapClient()
    return _client


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
            "source": "coincap",
        }
    except Exception as e:
        print(f"[CoinCap] API failed: {e}, using mock data", file=sys.stderr)
        return _get_mock_price(symbol)


def _get_mock_history(symbol: str, start_dt: datetime, end_dt: datetime) -> list:
    """Return mock historical data when API is unavailable"""
    import hashlib

    base_prices = {"BTC": 97000, "ETH": 3400, "SOL": 190}
    base_price = base_prices.get(symbol.upper(), 97000)

    data_points = []
    days = (end_dt - start_dt).days + 1

    for i in range(days):
        date = start_dt + timedelta(days=i)
        # Use deterministic hash based on date and symbol for consistent results
        seed_str = f"{symbol}{date.strftime('%Y-%m-%d')}"
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        # Generate price variation between -5% and +5% based on hash
        variation = ((hash_val % 10000) / 10000 - 0.5) * 0.1
        price = base_price * (1 + variation)

        data_points.append({
            "timestamp": date.isoformat(),
            "price_usd": round(price, 2),
        })

    return data_points


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
        days: Number of days (1-365), used if start_date/end_date not provided
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict with symbol, interval, data list, start_date, end_date
    """
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

        source = "coincap"
    except Exception as e:
        print(f"[CoinCap] History API failed: {e}, using mock data", file=sys.stderr)
        data_points = _get_mock_history(symbol, start_dt, end_dt)
        source = "mock"

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "data": data_points,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "source": source,
    }
