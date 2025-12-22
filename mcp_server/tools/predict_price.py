#!/usr/bin/env python3
"""
Crypto Price Predictor - MCP Tool Server (stdio mode)

This server provides cryptocurrency price prediction tools via MCP protocol.
It communicates using JSON-RPC over stdio (stdin/stdout).

Usage:
    python predict_price.py

Tools:
    - predict_price_price_only: Predict using price data only
    - predict_price_with_text: Predict using price data + news sentiment
"""

import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP

from config import env
from data import get_price_data
from features import get_latest_features, get_price_text_features, PRICE_ONLY_FEATURES, PRICE_TEXT_FEATURES
from models import get_predictor, get_text_predictor, get_transformer_predictor, TRANSFORMER_SEQUENCE_LENGTH
from api import get_current_price as api_get_current_price, get_price_history as api_get_price_history
from charts import generate_chart
from news import analyze_news_full

from pydantic import BaseModel, field_validator
from typing import Literal


# ===== Input/Output Schemas =====

# ----- Price Query Schemas -----

class GetPriceInput(BaseModel):
    """Input schema for get_current_price tool"""
    symbol: Literal["BTC", "ETH", "SOL"]


class GetPriceOutput(BaseModel):
    """Output schema for get_current_price tool"""
    symbol: str
    price_usd: float
    change_24h_percent: float
    volume_24h_usd: float
    market_cap_usd: float
    timestamp: str
    source: str


class GetHistoryInput(BaseModel):
    """Input schema for get_price_history tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    interval: Literal["h1", "h2", "h6", "h12", "d1"] = "d1"
    days: int | None = None
    start_date: str | None = None  # YYYY-MM-DD format
    end_date: str | None = None    # YYYY-MM-DD format

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: int | None) -> int | None:
        if v is not None and not 1 <= v <= 365:
            raise ValueError("days must be between 1 and 365")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("date must be in YYYY-MM-DD format")
        return v


class PricePoint(BaseModel):
    """Single price data point"""
    timestamp: str
    price_usd: float


class GetHistoryOutput(BaseModel):
    """Output schema for get_price_history tool"""
    symbol: str
    interval: str
    data: list[PricePoint]
    start_date: str
    end_date: str
    source: str


# ----- Chart Schemas -----

class GenerateChartInput(BaseModel):
    """Input schema for generate_price_chart tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    chart_type: Literal["line", "candlestick", "area"] = "line"
    days: int = 30
    include_volume: bool = True
    indicators: list[Literal["sma_20", "sma_50", "bollinger"]] = []

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: int) -> int:
        if not 1 <= v <= 365:
            raise ValueError("days must be between 1 and 365")
        return v


class GenerateChartOutput(BaseModel):
    """Output schema for generate_price_chart tool"""
    symbol: str
    chart_type: str
    image_base64: str
    image_format: str
    days: int
    indicators: list[str]


# ----- Prediction Schemas -----

class PredictPriceInput(BaseModel):
    """Input schema for predict_price_price_only tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    as_of: str  # YYYY-MM-DD format or "now"

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, v: str) -> str:
        if v == "now":
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("as_of must be 'now' or YYYY-MM-DD format")
        return v


class PredictPriceOutput(BaseModel):
    """Output schema for predict_price_price_only tool"""
    symbol: str
    as_of: str
    prediction: Literal["UP", "DOWN"]
    prob_up: float
    model_variant: str
    features_used: list[str]


class PredictPriceWithTextInput(BaseModel):
    """Input schema for predict_price_with_text tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    as_of: str
    news_text: str

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, v: str) -> str:
        if v == "now":
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("as_of must be 'now' or YYYY-MM-DD format")
        return v

    @field_validator("news_text")
    @classmethod
    def validate_news_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("news_text cannot be empty")
        return v.strip()


class PredictPriceWithTextOutput(BaseModel):
    """Output schema for predict_price_with_text tool"""
    symbol: str
    as_of: str
    prediction: Literal["UP", "DOWN"]
    prob_up: float
    model_variant: str
    features_used: list[str]
    sentiment_score: float


# ----- Sentiment Analysis Schemas -----

class AnalyzeSentimentInput(BaseModel):
    """Input schema for analyze_news_sentiment tool"""
    news_text: str
    include_keywords: bool = True

    @field_validator("news_text")
    @classmethod
    def validate_news_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("news_text cannot be empty")
        if len(v) > 5000:
            raise ValueError("news_text exceeds maximum length of 5000 characters")
        return v.strip()


class AnalyzeSentimentOutput(BaseModel):
    """Output schema for analyze_news_sentiment tool"""
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    score: float
    confidence: float
    key_phrases: list[str]
    impact_level: Literal["HIGH", "MEDIUM", "LOW"]


# ----- Transformer Prediction Schemas -----

class TransformerPredictInput(BaseModel):
    """Input schema for predict_price_transformer tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    prediction_horizon: Literal["1d", "3d", "7d"] = "1d"


class ConfidenceInterval(BaseModel):
    """Confidence interval for prediction"""
    lower: float
    upper: float
    confidence_level: float


class TransformerPredictOutput(BaseModel):
    """Output schema for predict_price_transformer tool"""
    symbol: str
    prediction: Literal["UP", "DOWN"]
    prob_up: float
    prediction_horizon: str
    confidence_interval: ConfidenceInterval
    attention_summary: dict
    model_info: dict


# ===== MCP Server =====

mcp = FastMCP("crypto-predictor")


@mcp.tool()
def predict_price_price_only(input: PredictPriceInput) -> PredictPriceOutput:
    """
    Predict next-day cryptocurrency price movement using price data only.

    Uses historical OHLCV price data to predict whether the price will go UP or DOWN.

    Features used:
    - return_1d: 1-day price return
    - rsi_14: 14-day Relative Strength Index
    - volatility_7d: 7-day rolling volatility

    Args:
        input: Symbol (BTC/ETH/SOL) and reference date (YYYY-MM-DD or "now")

    Returns:
        Prediction result with probability and model metadata
    """
    # Resolve "now" to actual date
    if input.as_of == "now":
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    else:
        as_of_date = input.as_of

    # Log to stderr (stdout is for MCP protocol)
    print(f"[Tool] predict_price_price_only: {input.symbol} @ {as_of_date}", file=sys.stderr)

    # Get price data
    df = get_price_data(
        symbol=input.symbol,
        as_of=as_of_date,
        lookback_days=60,
    )

    # Build features
    features = get_latest_features(df)
    print(f"  Features: {features}", file=sys.stderr)

    # Get prediction
    predictor = get_predictor()
    prediction, prob_up = predictor.predict(features)

    print(f"  Prediction: {prediction} (prob_up={prob_up:.4f})", file=sys.stderr)

    return PredictPriceOutput(
        symbol=input.symbol,
        as_of=as_of_date,
        prediction=prediction,
        prob_up=round(prob_up, 4),
        model_variant="price_only",
        features_used=PRICE_ONLY_FEATURES,
    )


@mcp.tool()
def predict_price_with_text(input: PredictPriceWithTextInput) -> PredictPriceWithTextOutput:
    """
    Predict next-day cryptocurrency price movement using price data AND news sentiment.

    Combines historical price data with sentiment analysis of news text.

    Features used:
    - return_1d: 1-day price return
    - rsi_14: 14-day RSI
    - volatility_7d: 7-day volatility
    - sentiment_score: News sentiment (-1.0 to 1.0)

    Args:
        input: Symbol, reference date, and news text for sentiment analysis

    Returns:
        Prediction result with probability, sentiment score, and model metadata
    """
    # Resolve "now" to actual date
    if input.as_of == "now":
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    else:
        as_of_date = input.as_of

    print(f"[Tool] predict_price_with_text: {input.symbol} @ {as_of_date}", file=sys.stderr)
    print(f"  News text: {input.news_text[:50]}...", file=sys.stderr)

    # Get price data
    df = get_price_data(
        symbol=input.symbol,
        as_of=as_of_date,
        lookback_days=60,
    )

    # Build combined features (price + text)
    features = get_price_text_features(df, input.news_text)
    print(f"  Features: {features}", file=sys.stderr)

    # Get prediction from price+text model
    predictor = get_text_predictor()
    prediction, prob_up = predictor.predict(features)

    print(f"  Prediction: {prediction} (prob_up={prob_up:.4f})", file=sys.stderr)

    return PredictPriceWithTextOutput(
        symbol=input.symbol,
        as_of=as_of_date,
        prediction=prediction,
        prob_up=round(prob_up, 4),
        model_variant="price_plus_text",
        features_used=PRICE_TEXT_FEATURES,
        sentiment_score=features["sentiment_score"],
    )


# ===== Price Query Tools =====

@mcp.tool()
def get_current_price(input: GetPriceInput) -> GetPriceOutput:
    """
    Get real-time cryptocurrency price from CoinCap API.

    Returns current price in USD, 24-hour change percentage,
    24-hour trading volume, and market capitalization.

    Data source: CoinCap API (no rate limits, real-time data)

    Args:
        input: Symbol to query (BTC, ETH, or SOL)

    Returns:
        Current price data with market statistics
    """
    print(f"[Tool] get_current_price: {input.symbol}", file=sys.stderr)

    try:
        data = api_get_current_price(input.symbol)
        print(f"  Price: ${data['price_usd']:.2f}", file=sys.stderr)
        return GetPriceOutput(**data)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        raise


@mcp.tool()
def get_price_history(input: GetHistoryInput) -> GetHistoryOutput:
    """
    Get historical cryptocurrency price data from CoinCap API.

    Returns price history for the specified time period and interval.
    Useful for charting, technical analysis, and trend identification.

    Intervals:
    - h1: 1-hour candles
    - h2: 2-hour candles
    - h6: 6-hour candles
    - h12: 12-hour candles
    - d1: Daily candles (default)

    Args:
        input: Symbol, interval (h1/h2/h6/h12/d1), and number of days (1-365)
               OR start_date and end_date in YYYY-MM-DD format

    Returns:
        List of price points with timestamps
    """
    if input.start_date and input.end_date:
        print(f"[Tool] get_price_history: {input.symbol} ({input.interval}, {input.start_date} ~ {input.end_date})", file=sys.stderr)
    else:
        print(f"[Tool] get_price_history: {input.symbol} ({input.interval}, {input.days or 30} days)", file=sys.stderr)

    try:
        data = api_get_price_history(
            input.symbol,
            input.interval,
            days=input.days,
            start_date=input.start_date,
            end_date=input.end_date
        )
        print(f"  Retrieved {len(data['data'])} data points", file=sys.stderr)

        # Convert to Pydantic models
        price_points = [PricePoint(**p) for p in data["data"]]

        return GetHistoryOutput(
            symbol=data["symbol"],
            interval=data["interval"],
            data=price_points,
            start_date=data["start_date"],
            end_date=data["end_date"],
            source=data["source"],
        )
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        raise


@mcp.tool()
def generate_price_chart(input: GenerateChartInput) -> GenerateChartOutput:
    """
    Generate a price chart as a PNG image (base64 encoded).

    Creates visual price charts with optional technical indicators.
    Returns base64-encoded PNG that can be displayed in web browsers.

    Chart Types:
    - line: Simple line chart with area fill
    - candlestick: OHLC candlestick chart (requires mplfinance)
    - area: Filled area chart

    Indicators:
    - sma_20: 20-day Simple Moving Average
    - sma_50: 50-day Simple Moving Average
    - bollinger: Bollinger Bands (20-day, 2 std)

    Args:
        input: Symbol, chart type, days, and indicators

    Returns:
        Base64-encoded PNG image
    """
    print(f"[Tool] generate_price_chart: {input.symbol} ({input.chart_type}, {input.days} days)", file=sys.stderr)
    print(f"  Indicators: {input.indicators}", file=sys.stderr)

    try:
        # Get price data (use mock data for OHLCV)
        df = get_price_data(
            symbol=input.symbol,
            as_of=datetime.now().strftime("%Y-%m-%d"),
            lookback_days=input.days,
        )

        # Generate chart
        image_base64 = generate_chart(
            df=df,
            symbol=input.symbol,
            chart_type=input.chart_type,
            include_volume=input.include_volume,
            indicators=list(input.indicators),
        )

        print(f"  Generated chart: {len(image_base64)} bytes (base64)", file=sys.stderr)

        return GenerateChartOutput(
            symbol=input.symbol,
            chart_type=input.chart_type,
            image_base64=image_base64,
            image_format="png",
            days=input.days,
            indicators=list(input.indicators),
        )
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        raise


@mcp.tool()
def analyze_news_sentiment(input: AnalyzeSentimentInput) -> AnalyzeSentimentOutput:
    """
    Analyze news text for cryptocurrency market sentiment.

    Uses FinBERT (financial BERT) when available, with rule-based fallback.
    Extracts key phrases and assesses potential market impact.

    Sentiment levels:
    - BULLISH: Positive sentiment, suggests price may rise
    - BEARISH: Negative sentiment, suggests price may fall
    - NEUTRAL: Mixed or no clear sentiment

    Impact levels:
    - HIGH: Major news (ETF approval, exchange hack, etc.)
    - MEDIUM: Moderate market-moving news
    - LOW: Minor or neutral news

    Args:
        input: News text and whether to include keyword analysis

    Returns:
        Sentiment analysis with score, confidence, key phrases, and impact level
    """
    print(f"[Tool] analyze_news_sentiment: {len(input.news_text)} chars", file=sys.stderr)
    print(f"  Preview: {input.news_text[:80]}...", file=sys.stderr)

    try:
        # Perform full analysis
        result = analyze_news_full(input.news_text)

        print(f"  Sentiment: {result['sentiment']} (score={result['score']:.4f})", file=sys.stderr)
        print(f"  Impact: {result['impact_level']}, Confidence: {result['confidence']}", file=sys.stderr)

        # Optionally exclude keywords
        key_phrases = result["key_phrases"] if input.include_keywords else []

        return AnalyzeSentimentOutput(
            sentiment=result["sentiment"],
            score=result["score"],
            confidence=result["confidence"],
            key_phrases=key_phrases,
            impact_level=result["impact_level"],
        )
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        raise


@mcp.tool()
def predict_price_transformer(input: TransformerPredictInput) -> TransformerPredictOutput:
    """
    Predict cryptocurrency price movement using Transformer deep learning model.

    Uses 60 days of OHLCV data with attention mechanism to capture
    temporal patterns and dependencies in price movements.

    Architecture:
    - Input: 60 days × 5 features (OHLCV)
    - Transformer Encoder with 2 layers, 4 attention heads
    - Output: UP/DOWN prediction with probability

    Falls back to statistical method (momentum + RSI) if PyTorch is unavailable.

    Args:
        input: Symbol (BTC/ETH/SOL) and prediction horizon (1d/3d/7d)

    Returns:
        Prediction with probability, confidence interval, and attention summary
    """
    print(f"[Tool] predict_price_transformer: {input.symbol} ({input.prediction_horizon})", file=sys.stderr)

    try:
        # Get price data (need 60+ days)
        df = get_price_data(
            symbol=input.symbol,
            as_of=datetime.now().strftime("%Y-%m-%d"),
            lookback_days=TRANSFORMER_SEQUENCE_LENGTH + 10,  # Extra buffer
        )

        print(f"  Data shape: {df.shape}", file=sys.stderr)

        # Get predictor
        predictor = get_transformer_predictor()

        # Make prediction
        prediction, prob_up, attention_summary = predictor.predict(df)

        print(f"  Prediction: {prediction} (prob_up={prob_up:.4f})", file=sys.stderr)
        print(f"  Method: {attention_summary.get('method', 'unknown')}", file=sys.stderr)

        # Calculate confidence interval based on probability
        # Higher probability = narrower interval
        margin = 0.1 * (1 - abs(prob_up - 0.5) * 2)  # 0.05 to 0.10
        confidence_interval = ConfidenceInterval(
            lower=max(0, prob_up - margin),
            upper=min(1, prob_up + margin),
            confidence_level=0.95,
        )

        # Model info
        model_info = {
            "method": attention_summary.get("method", "unknown"),
            "sequence_length": TRANSFORMER_SEQUENCE_LENGTH,
            "prediction_horizon": input.prediction_horizon,
            "features": ["open", "high", "low", "close", "volume"],
        }

        return TransformerPredictOutput(
            symbol=input.symbol,
            prediction=prediction,
            prob_up=round(prob_up, 4),
            prediction_horizon=input.prediction_horizon,
            confidence_interval=confidence_interval,
            attention_summary=attention_summary,
            model_info=model_info,
        )
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        raise


def main():
    """Start the MCP server in stdio mode."""
    print("=" * 50, file=sys.stderr)
    print("Crypto Predictor MCP Server (stdio mode)", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # Pre-load models
    print("[Startup] Loading models...", file=sys.stderr)
    try:
        predictor = get_predictor()
        print("[Startup] Price-only model loaded", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}", file=sys.stderr)
        print("[Startup] Run 'python train_model.py' first", file=sys.stderr)
        sys.exit(1)

    try:
        text_predictor = get_text_predictor()
        print("[Startup] Price+text model loaded", file=sys.stderr)
    except FileNotFoundError:
        print("[Startup] Price+text model not found, skipping", file=sys.stderr)

    print("[Server] MCP server running on stdio", file=sys.stderr)
    print("[Server] Tools:", file=sys.stderr)
    print("  - get_current_price: Real-time price query", file=sys.stderr)
    print("  - get_price_history: Historical price data", file=sys.stderr)
    print("  - generate_price_chart: Price chart generation", file=sys.stderr)
    print("  - analyze_news_sentiment: News sentiment analysis", file=sys.stderr)
    print("  - predict_price_price_only: LightGBM prediction", file=sys.stderr)
    print("  - predict_price_with_text: LightGBM + sentiment", file=sys.stderr)
    print("  - predict_price_transformer: Transformer deep learning", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # Run the server (stdio mode is default in FastMCP)
    mcp.run()


if __name__ == "__main__":
    main()
