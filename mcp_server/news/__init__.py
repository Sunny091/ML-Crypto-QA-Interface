"""
News Analysis Module - Phase 4

Provides sentiment analysis for news text.
Supports:
- FinBERT (transformer-based, accurate)
- Mock (rule-based, fast, for testing)

All analysis is deterministic - same input always produces same output.
"""

import re
import hashlib
from typing import Literal

from config import env


def _mock_sentiment(text: str) -> float:
    """
    Rule-based mock sentiment analyzer.
    Deterministic: same text always returns same score.

    Returns:
        sentiment_score in range [-1.0, 1.0]
    """
    text_lower = text.lower()

    # Positive keywords (crypto/financial context)
    positive_words = [
        'surge', 'soar', 'rally', 'bullish', 'gain', 'rise', 'jump',
        'breakthrough', 'adoption', 'institutional', 'etf', 'approval',
        'growth', 'profit', 'milestone', 'record', 'high', 'buy',
        '上漲', '突破', '利好', '看漲', '成長', '創新高', '買入',
        '漲', '好', '強', '增', '升'
    ]

    # Negative keywords
    negative_words = [
        'crash', 'plunge', 'bearish', 'drop', 'fall', 'decline', 'sell',
        'ban', 'regulation', 'hack', 'fraud', 'scam', 'collapse', 'risk',
        'loss', 'dump', 'fear', 'uncertainty', 'lawsuit', 'sec',
        '下跌', '暴跌', '利空', '看跌', '禁止', '監管', '風險',
        '跌', '壞', '弱', '減', '降'
    ]

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    total = positive_count + negative_count
    if total == 0:
        # Use hash for deterministic neutral score with slight variation
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return (hash_val % 200 - 100) / 500  # Range: [-0.2, 0.2]

    # Calculate weighted sentiment
    score = (positive_count - negative_count) / total

    # Scale to [-1, 1] with some smoothing
    return max(-1.0, min(1.0, score * 0.8))


def _finbert_sentiment(text: str) -> float:
    """
    FinBERT-based sentiment analysis.
    Falls back to mock if transformers not available.

    Returns:
        sentiment_score in range [-1.0, 1.0]
    """
    try:
        from transformers import pipeline

        # Use cached classifier
        if not hasattr(_finbert_sentiment, '_classifier'):
            print("[News] Loading FinBERT model...")
            _finbert_sentiment._classifier = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,  # Return all scores
            )
            print("[News] FinBERT model loaded!")

        classifier = _finbert_sentiment._classifier

        # Truncate text to max length
        truncated = text[:env.news.max_length]

        # Get predictions
        results = classifier(truncated)[0]

        # Convert to single score
        # FinBERT outputs: positive, negative, neutral
        score_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

        weighted_score = 0.0
        for item in results:
            label = item['label'].lower()
            weight = score_map.get(label, 0.0)
            weighted_score += weight * item['score']

        return max(-1.0, min(1.0, weighted_score))

    except ImportError:
        print("[News] transformers not available, falling back to mock sentiment")
        return _mock_sentiment(text)
    except Exception as e:
        print(f"[News] FinBERT error: {e}, falling back to mock sentiment")
        return _mock_sentiment(text)


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of news text.

    Args:
        text: News article text (max length from config)

    Returns:
        sentiment_score in range [-1.0, 1.0]
        - Positive values indicate bullish sentiment
        - Negative values indicate bearish sentiment
        - Zero indicates neutral sentiment
    """
    if not text or not text.strip():
        return 0.0

    # Truncate to max length
    text = text[:env.news.max_length]

    # Choose analyzer based on config
    if env.news.model_type == "finbert":
        return _finbert_sentiment(text)
    else:
        return _mock_sentiment(text)


# Convenience function for feature extraction
def get_news_features(news_text: str | None) -> dict[str, float]:
    """
    Extract news features for ML model.

    Args:
        news_text: Optional news text

    Returns:
        Dictionary with:
        - sentiment_score: float in [-1, 1]
        - news_present: 0.0 or 1.0
    """
    if news_text is None or not news_text.strip():
        return {
            "sentiment_score": 0.0,
            "news_present": 0.0,
        }

    return {
        "sentiment_score": round(analyze_sentiment(news_text), 4),
        "news_present": 1.0,
    }
