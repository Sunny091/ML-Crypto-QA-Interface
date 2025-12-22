#!/usr/bin/env python3
"""
MCP Tool Server for cryptocurrency price prediction.

Phase 2: Execution Plane only.
- Provides deterministic tool execution
- No natural language understanding
- No connection to Orchestrator (Phase 3)

Usage:
    python server.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP

from config import env
from data import get_price_data
from features import get_latest_features, PRICE_ONLY_FEATURES
from models import get_predictor
from tools import PredictPriceInput, PredictPriceOutput


# Initialize FastMCP server
mcp = FastMCP(env.mcp.server_name)


@mcp.tool()
def predict_price_price_only(input: PredictPriceInput) -> PredictPriceOutput:
    """
    Predict next-day price movement using price data only.

    This tool uses historical OHLCV price data to predict whether
    the price will go UP or DOWN the next day.

    Features used:
    - return_1d: 1-day price return
    - rsi_14: 14-day Relative Strength Index
    - volatility_7d: 7-day rolling volatility

    Args:
        input: Symbol and reference date for prediction

    Returns:
        Prediction result with probability and model metadata
    """
    # Resolve "now" to actual date
    if input.as_of == "now":
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    else:
        as_of_date = input.as_of

    print(f"[Tool] predict_price_price_only called")
    print(f"  - Symbol: {input.symbol}")
    print(f"  - As of: {as_of_date}")

    # Get price data
    df = get_price_data(
        symbol=input.symbol,
        as_of=as_of_date,
        lookback_days=60,
    )

    # Build features
    features = get_latest_features(df)
    print(f"  - Features: {features}")

    # Get prediction
    predictor = get_predictor()
    prediction, prob_up = predictor.predict(features)

    print(f"  - Prediction: {prediction} (prob_up={prob_up:.4f})")

    return PredictPriceOutput(
        symbol=input.symbol,
        as_of=as_of_date,
        prediction=prediction,
        prob_up=round(prob_up, 4),
        model_variant="price_only",
        features_used=PRICE_ONLY_FEATURES,
    )


def main():
    """Start the MCP server."""
    print("")
    print("=" * 60)
    print("MCP Tool Server - Phase 2: Execution Plane")
    print("=" * 60)
    print(f"  Server Name: {env.mcp.server_name}")
    print(f"  Host:        {env.mcp.server_host}")
    print(f"  Port:        {env.mcp.server_port}")
    print(f"  Environment: {env.app.app_env}")
    print("=" * 60)
    print("")

    # Pre-load model
    print("[Startup] Loading model...")
    try:
        predictor = get_predictor()
        print("[Startup] Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] Run 'python train_model.py' first to create the model.")
        sys.exit(1)

    print("")
    print("[Server] Starting MCP server...")
    print("[Server] Tools available:")
    print("  - predict_price_price_only")
    print("")

    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
