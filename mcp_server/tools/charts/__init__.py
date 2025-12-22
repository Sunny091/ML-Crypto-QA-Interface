"""
Chart Generation Module

Generates cryptocurrency price charts using matplotlib.
Returns base64-encoded PNG images.
"""

import io
import base64
import sys
from typing import Literal
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Try to import mplfinance for candlestick charts
try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False
    print("[Charts] mplfinance not installed, candlestick charts will use line fallback", file=sys.stderr)


# Dark theme colors matching the frontend
COLORS = {
    'background': '#1a1a2e',
    'panel': '#16213e',
    'line': '#e94560',
    'up': '#4ade80',
    'down': '#f87171',
    'text': '#eeeeee',
    'grid': '#333333',
    'sma20': '#4ade80',
    'sma50': '#60a5fa',
    'bollinger': '#fbbf24',
}


def _setup_dark_style():
    """Configure matplotlib for dark theme"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
    })


def _add_indicators(ax, df: pd.DataFrame, indicators: list[str]):
    """Add technical indicators to chart"""
    close = df['close'] if 'close' in df.columns else df['price_usd']
    dates = df['date'] if 'date' in df.columns else df.index

    if 'sma_20' in indicators:
        sma20 = close.rolling(20).mean()
        ax.plot(dates, sma20, '--', color=COLORS['sma20'], label='SMA 20', linewidth=1)

    if 'sma_50' in indicators:
        sma50 = close.rolling(50).mean()
        ax.plot(dates, sma50, '--', color=COLORS['sma50'], label='SMA 50', linewidth=1)

    if 'bollinger' in indicators:
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        ax.fill_between(dates, sma - 2*std, sma + 2*std,
                        alpha=0.2, color=COLORS['bollinger'], label='Bollinger Bands')

    if indicators:
        ax.legend(loc='upper left', fontsize=8)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100,
                facecolor=COLORS['background'], edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64


def generate_line_chart(
    df: pd.DataFrame,
    symbol: str,
    title: str = None,
    include_volume: bool = True,
    indicators: list[str] = None,
) -> str:
    """
    Generate a line chart with optional volume and indicators.

    Args:
        df: DataFrame with columns: date, close (or price_usd), and optionally volume
        symbol: Cryptocurrency symbol for title
        title: Custom title (optional)
        include_volume: Whether to include volume subplot
        indicators: List of indicators to add (sma_20, sma_50, bollinger)

    Returns:
        Base64-encoded PNG image
    """
    _setup_dark_style()
    indicators = indicators or []

    # Prepare data
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp'])

    close = df['close'] if 'close' in df.columns else df['price_usd']
    dates = df['date'] if 'date' in df.columns else df.index

    has_volume = include_volume and 'volume' in df.columns

    # Create figure
    if has_volume:
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=(12, 6))

    # Price line
    ax_price.plot(dates, close, color=COLORS['line'], linewidth=2)
    ax_price.fill_between(dates, close, alpha=0.1, color=COLORS['line'])

    # Add indicators
    _add_indicators(ax_price, df, indicators)

    # Labels
    ax_price.set_title(title or f'{symbol} Price Chart', fontsize=14, fontweight='bold')
    ax_price.set_ylabel('Price (USD)', fontsize=10)
    ax_price.grid(True)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Volume subplot
    if has_volume:
        if 'open' in df.columns:
            colors = [COLORS['up'] if c >= o else COLORS['down']
                      for c, o in zip(df['close'], df['open'])]
        else:
            colors = [COLORS['up']] * len(df)

        ax_vol.bar(dates, df['volume'], color=colors, alpha=0.7, width=0.8)
        ax_vol.set_ylabel('Volume', fontsize=10)
        ax_vol.grid(True)

    # Format x-axis
    ax = ax_vol if has_volume else ax_price
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    include_volume: bool = True,
    indicators: list[str] = None,
) -> str:
    """
    Generate a candlestick (OHLC) chart.

    Args:
        df: DataFrame with columns: date, open, high, low, close, and optionally volume
        symbol: Cryptocurrency symbol for title
        include_volume: Whether to include volume subplot
        indicators: List of indicators to add

    Returns:
        Base64-encoded PNG image
    """
    if not HAS_MPLFINANCE:
        # Fallback to line chart
        print("[Charts] Using line chart fallback for candlestick", file=sys.stderr)
        return generate_line_chart(df, symbol, f'{symbol} Price Chart',
                                   include_volume, indicators)

    indicators = indicators or []

    # Prepare data for mplfinance
    df_mpf = df.copy()
    if 'date' in df_mpf.columns:
        df_mpf = df_mpf.set_index('date')
    elif 'timestamp' in df_mpf.columns:
        df_mpf['date'] = pd.to_datetime(df_mpf['timestamp'])
        df_mpf = df_mpf.set_index('date')

    # Rename columns for mplfinance
    df_mpf = df_mpf.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close']
    if not all(col in df_mpf.columns for col in required):
        print("[Charts] Missing OHLC columns, using line chart fallback", file=sys.stderr)
        return generate_line_chart(df, symbol, f'{symbol} Price Chart',
                                   include_volume, indicators)

    # Custom market colors
    mc = mpf.make_marketcolors(
        up=COLORS['up'], down=COLORS['down'],
        edge='inherit', wick='inherit',
        volume='inherit'
    )

    # Custom style
    s = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor=COLORS['panel'],
        gridcolor=COLORS['grid'],
        gridstyle=':',
        rc={
            'figure.facecolor': COLORS['background'],
            'axes.labelcolor': COLORS['text'],
            'axes.edgecolor': COLORS['grid'],
            'xtick.color': COLORS['text'],
            'ytick.color': COLORS['text'],
        }
    )

    # Add indicator plots
    add_plots = []
    if 'sma_20' in indicators and len(df_mpf) >= 20:
        sma20 = df_mpf['Close'].rolling(20).mean()
        add_plots.append(mpf.make_addplot(sma20, color=COLORS['sma20'], width=1))
    if 'sma_50' in indicators and len(df_mpf) >= 50:
        sma50 = df_mpf['Close'].rolling(50).mean()
        add_plots.append(mpf.make_addplot(sma50, color=COLORS['sma50'], width=1))

    # Generate chart
    buffer = io.BytesIO()

    mpf.plot(
        df_mpf, type='candle', style=s,
        volume=include_volume and 'Volume' in df_mpf.columns,
        addplot=add_plots if add_plots else None,
        title=f'{symbol} Price Chart',
        figsize=(12, 8),
        savefig=dict(fname=buffer, dpi=100, facecolor=COLORS['background'], format='png')
    )

    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return image_base64


def generate_area_chart(
    df: pd.DataFrame,
    symbol: str,
    include_volume: bool = False,
    indicators: list[str] = None,
) -> str:
    """
    Generate an area chart (filled line chart).

    Args:
        df: DataFrame with columns: date/timestamp and close/price_usd
        symbol: Cryptocurrency symbol for title
        include_volume: Whether to include volume subplot
        indicators: List of indicators to add

    Returns:
        Base64-encoded PNG image
    """
    _setup_dark_style()
    indicators = indicators or []

    # Prepare data
    df = df.copy()
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])

    close = df['close'] if 'close' in df.columns else df['price_usd']
    dates = df['date'] if 'date' in df.columns else df.index

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Area fill
    ax.fill_between(dates, close, alpha=0.4, color=COLORS['line'])
    ax.plot(dates, close, color=COLORS['line'], linewidth=2)

    # Add indicators
    _add_indicators(ax, df, indicators)

    # Labels
    ax.set_title(f'{symbol} Price Chart', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=10)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_chart(
    df: pd.DataFrame,
    symbol: str,
    chart_type: Literal["line", "candlestick", "area"] = "line",
    include_volume: bool = True,
    indicators: list[str] = None,
) -> str:
    """
    Generate a price chart of the specified type.

    Args:
        df: DataFrame with price data
        symbol: Cryptocurrency symbol
        chart_type: Type of chart (line, candlestick, area)
        include_volume: Whether to include volume
        indicators: List of indicators (sma_20, sma_50, bollinger)

    Returns:
        Base64-encoded PNG image
    """
    if chart_type == "candlestick":
        return generate_candlestick_chart(df, symbol, include_volume, indicators)
    elif chart_type == "area":
        return generate_area_chart(df, symbol, include_volume, indicators)
    else:
        return generate_line_chart(df, symbol, None, include_volume, indicators)
