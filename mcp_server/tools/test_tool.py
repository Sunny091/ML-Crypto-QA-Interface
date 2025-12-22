#!/usr/bin/env python3
"""
Test script for the predict_price_price_only tool.

This script tests the tool directly without going through MCP protocol.
Use this to verify the pipeline works before starting the server.

Usage:
    python test_tool.py
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import env
from data import get_price_data
from features import get_latest_features, build_features, PRICE_ONLY_FEATURES
from models import get_predictor
from tools import PredictPriceInput, PredictPriceOutput


def test_data_generation():
    """Test mock data generation."""
    print("\n" + "=" * 50)
    print("Test 1: Mock Data Generation")
    print("=" * 50)

    for symbol in ["BTC", "ETH", "SOL"]:
        df = get_price_data(symbol, as_of="2024-06-15", lookback_days=30)
        print(f"\n{symbol}:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  - Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    print("\n[PASS] Data generation works!")


def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "=" * 50)
    print("Test 2: Feature Engineering")
    print("=" * 50)

    df = get_price_data("BTC", as_of="2024-06-15", lookback_days=30)
    df_with_features = build_features(df)

    print(f"\nFeature columns added: {PRICE_ONLY_FEATURES}")
    print(f"\nSample of last 5 rows:")
    print(df_with_features[["date", "close"] + PRICE_ONLY_FEATURES].tail())

    features = get_latest_features(df)
    print(f"\nLatest features: {features}")

    print("\n[PASS] Feature engineering works!")


def test_model_prediction():
    """Test model loading and prediction."""
    print("\n" + "=" * 50)
    print("Test 3: Model Prediction")
    print("=" * 50)

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        print(f"\n[SKIP] {e}")
        print("Run 'python train_model.py' first to create the model.")
        return False

    # Test with sample features
    features = {
        "return_1d": 0.02,
        "rsi_14": 65.0,
        "volatility_7d": 0.03,
    }

    prediction, prob_up = predictor.predict(features)
    print(f"\nTest features: {features}")
    print(f"Prediction: {prediction}")
    print(f"Probability UP: {prob_up:.4f}")

    print("\n[PASS] Model prediction works!")
    return True


def test_full_pipeline():
    """Test the complete pipeline."""
    print("\n" + "=" * 50)
    print("Test 4: Full Pipeline (Tool Simulation)")
    print("=" * 50)

    try:
        predictor = get_predictor()
    except FileNotFoundError:
        print("\n[SKIP] Model not trained yet.")
        return False

    test_cases = [
        {"symbol": "BTC", "as_of": "2024-06-15"},
        {"symbol": "ETH", "as_of": "2024-03-01"},
        {"symbol": "SOL", "as_of": "now"},
    ]

    for case in test_cases:
        print(f"\n--- Testing {case} ---")

        # Simulate tool execution
        input_data = PredictPriceInput(**case)

        # Get data
        as_of = input_data.as_of
        if as_of == "now":
            from datetime import datetime
            as_of = datetime.now().strftime("%Y-%m-%d")

        df = get_price_data(input_data.symbol, as_of=as_of, lookback_days=60)
        features = get_latest_features(df)
        prediction, prob_up = predictor.predict(features)

        output = PredictPriceOutput(
            symbol=input_data.symbol,
            as_of=as_of,
            prediction=prediction,
            prob_up=round(prob_up, 4),
            model_variant="price_only",
            features_used=PRICE_ONLY_FEATURES,
        )

        print(f"Input:  {input_data.model_dump()}")
        print(f"Output: {output.model_dump()}")

    print("\n[PASS] Full pipeline works!")
    return True


def main():
    print("=" * 50)
    print("MCP Tool Server - Test Suite")
    print("=" * 50)

    # Run tests
    test_data_generation()
    test_feature_engineering()

    model_ok = test_model_prediction()
    if model_ok:
        test_full_pipeline()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

    if not model_ok:
        print("\nNote: To run all tests, first train the model:")
        print("  python train_model.py")


if __name__ == "__main__":
    main()
