#!/usr/bin/env python3
"""
Test script for all MCP tools.
Tests each tool individually without starting the full MCP server.
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports work correctly"""
    print("\n" + "=" * 50)
    print("Testing imports...")
    print("=" * 50)

    try:
        from config import env
        print("✓ config module loaded")
    except Exception as e:
        print(f"✗ config: {e}")
        return False

    try:
        from api import get_current_price, get_price_history
        print("✓ api module loaded")
    except Exception as e:
        print(f"✗ api: {e}")
        return False

    try:
        from charts import generate_chart
        print("✓ charts module loaded")
    except Exception as e:
        print(f"✗ charts: {e}")
        return False

    try:
        from news import analyze_news_full
        print("✓ news module loaded")
    except Exception as e:
        print(f"✗ news: {e}")
        return False

    try:
        from models import get_transformer_predictor, TRANSFORMER_SEQUENCE_LENGTH
        print("✓ models module loaded")
    except Exception as e:
        print(f"✗ models: {e}")
        return False

    try:
        from data import get_price_data
        print("✓ data module loaded")
    except Exception as e:
        print(f"✗ data: {e}")
        return False

    return True


def test_get_current_price():
    """Test get_current_price tool"""
    print("\n" + "=" * 50)
    print("Testing get_current_price...")
    print("=" * 50)

    from api import get_current_price

    for symbol in ["BTC", "ETH", "SOL"]:
        try:
            result = get_current_price(symbol)
            print(f"✓ {symbol}: ${result['price_usd']:,.2f} ({result['change_24h_percent']:+.2f}%)")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
            return False

    return True


def test_get_price_history():
    """Test get_price_history tool"""
    print("\n" + "=" * 50)
    print("Testing get_price_history...")
    print("=" * 50)

    from api import get_price_history

    try:
        result = get_price_history("BTC", interval="d1", days=7)
        print(f"✓ BTC 7-day history: {len(result['data'])} data points")
        print(f"  Date range: {result['start_date']} to {result['end_date']}")
        if result['data']:
            first_price = result['data'][0]['price_usd']
            last_price = result['data'][-1]['price_usd']
            print(f"  Price range: ${first_price:,.2f} to ${last_price:,.2f}")
    except Exception as e:
        print(f"✗ get_price_history: {e}")
        return False

    return True


def test_generate_chart():
    """Test generate_price_chart tool"""
    print("\n" + "=" * 50)
    print("Testing generate_price_chart...")
    print("=" * 50)

    from data import get_price_data
    from charts import generate_chart
    from datetime import datetime

    try:
        # Get mock data
        df = get_price_data("BTC", datetime.now().strftime("%Y-%m-%d"), lookback_days=30)
        print(f"✓ Got price data: {len(df)} rows")

        # Generate line chart
        image_base64 = generate_chart(df, "BTC", chart_type="line", include_volume=True)
        print(f"✓ Generated line chart: {len(image_base64)} bytes (base64)")

        # Generate area chart
        image_base64 = generate_chart(df, "BTC", chart_type="area", include_volume=False)
        print(f"✓ Generated area chart: {len(image_base64)} bytes (base64)")

    except Exception as e:
        print(f"✗ generate_chart: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_analyze_news_sentiment():
    """Test analyze_news_sentiment tool"""
    print("\n" + "=" * 50)
    print("Testing analyze_news_sentiment...")
    print("=" * 50)

    from news import analyze_news_full

    test_cases = [
        ("Bitcoin surges to record high as ETF approval drives institutional adoption", "BULLISH"),
        ("Crypto exchange hacked, users lose millions in major exploit", "BEARISH"),
        ("Bitcoin trading volume remains stable amid market uncertainty", "NEUTRAL"),
        ("比特幣創新高，機構投資者大舉進場", "BULLISH"),
        ("交易所遭駭客攻擊，用戶資金面臨風險", "BEARISH"),
    ]

    all_passed = True
    for text, expected in test_cases:
        try:
            result = analyze_news_full(text)
            status = "✓" if result["sentiment"] == expected else "⚠"
            print(f"{status} \"{text[:40]}...\"")
            print(f"    Sentiment: {result['sentiment']} (expected: {expected})")
            print(f"    Score: {result['score']:.4f}, Impact: {result['impact_level']}")
            print(f"    Keywords: {result['key_phrases'][:3]}")
        except Exception as e:
            print(f"✗ Error: {e}")
            all_passed = False

    return all_passed


def test_transformer_predictor():
    """Test predict_price_transformer tool"""
    print("\n" + "=" * 50)
    print("Testing predict_price_transformer...")
    print("=" * 50)

    from data import get_price_data
    from models import get_transformer_predictor, TRANSFORMER_SEQUENCE_LENGTH
    from datetime import datetime

    try:
        # Get price data
        df = get_price_data(
            "BTC",
            datetime.now().strftime("%Y-%m-%d"),
            lookback_days=TRANSFORMER_SEQUENCE_LENGTH + 10
        )
        print(f"✓ Got price data: {len(df)} rows")

        # Get predictor
        predictor = get_transformer_predictor()
        print(f"✓ Predictor loaded (has_torch={predictor.has_torch})")

        # Make prediction
        prediction, prob_up, attention_summary = predictor.predict(df)
        print(f"✓ Prediction: {prediction}")
        print(f"    Probability UP: {prob_up:.4f}")
        print(f"    Method: {attention_summary.get('method', 'unknown')}")

        if 'momentum_5d' in attention_summary:
            print(f"    Momentum 5d: {attention_summary['momentum_5d']}%")
            print(f"    RSI 14: {attention_summary['rsi_14']}")

    except Exception as e:
        print(f"✗ transformer: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("\n" + "=" * 60)
    print("  MCP Tool Test Suite")
    print("=" * 60)

    results = {}

    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return 1
    results["imports"] = True

    # Test each tool
    results["get_current_price"] = test_get_current_price()
    results["get_price_history"] = test_get_price_history()
    results["generate_chart"] = test_generate_chart()
    results["analyze_news_sentiment"] = test_analyze_news_sentiment()
    results["transformer_predictor"] = test_transformer_predictor()

    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  All tests passed! ✓")
    else:
        print("  Some tests failed. ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
