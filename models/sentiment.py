"""
Sentiment Analysis Module - 情感分析

Provides sentiment analysis for news text.
Supports:
- FinBERT (transformer-based, accurate)
- Mock (rule-based, fast, for testing)
"""

import re
import hashlib
from typing import Literal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import env


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
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return (hash_val % 200 - 100) / 500

    score = (positive_count - negative_count) / total
    return max(-1.0, min(1.0, score * 0.8))


def _finbert_sentiment(text: str) -> float:
    """
    FinBERT-based sentiment analysis.
    Falls back to mock if transformers not available.
    """
    try:
        from transformers import pipeline

        if not hasattr(_finbert_sentiment, '_classifier'):
            print("[News] Loading FinBERT model...")
            _finbert_sentiment._classifier = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
            )
            print("[News] FinBERT model loaded!")

        classifier = _finbert_sentiment._classifier
        truncated = text[:env.news.max_length]
        results = classifier(truncated)[0]

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
        text: News article text

    Returns:
        sentiment_score in range [-1.0, 1.0]
    """
    if not text or not text.strip():
        return 0.0

    text = text[:env.news.max_length]

    # Use mock for Chinese text
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars / max(len(text), 1) > 0.3:
        return _mock_sentiment(text)

    if env.news.model_type == "finbert":
        return _finbert_sentiment(text)
    else:
        return _mock_sentiment(text)


def get_news_features(news_text: str | None) -> dict[str, float]:
    """
    Extract news features for ML model.

    Args:
        news_text: Optional news text

    Returns:
        Dictionary with sentiment_score and news_present
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


# High-impact keywords
HIGH_IMPACT_KEYWORDS = {
    "positive": [
        "etf approved", "etf approval", "institutional adoption",
        "spot etf", "record high", "all-time high", "ath",
    ],
    "negative": [
        "sec lawsuit", "exchange hack", "major exploit", "bankruptcy",
        "exchange collapse", "ponzi scheme", "rug pull",
    ],
}


def analyze_news_full(text: str) -> dict:
    """
    Full news sentiment analysis with all details.

    Returns:
        Dictionary with sentiment, score, confidence, key_phrases, impact_level
    """
    if not text or not text.strip():
        return {
            "sentiment": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.0,
            "key_phrases": [],
            "impact_level": "LOW",
        }

    score = analyze_sentiment(text)

    # Determine sentiment label
    if score > 0.1:
        sentiment = "BULLISH"
    elif score < -0.1:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    # Extract key phrases
    text_lower = text.lower()
    found_phrases = []
    for category in ["positive", "negative"]:
        for phrase in HIGH_IMPACT_KEYWORDS[category]:
            if phrase in text_lower:
                found_phrases.append(phrase)

    # Assess impact
    abs_sentiment = abs(score)
    if any(phrase in text_lower for phrases in HIGH_IMPACT_KEYWORDS.values() for phrase in phrases):
        impact_level = "HIGH"
    elif abs_sentiment > 0.7:
        impact_level = "HIGH"
    elif abs_sentiment > 0.4:
        impact_level = "MEDIUM"
    else:
        impact_level = "LOW"

    # Calculate confidence
    confidence = 0.5 + (abs_sentiment * 0.3)
    if impact_level == "HIGH":
        confidence += 0.1
    if len(text) > 500:
        confidence += 0.05

    return {
        "sentiment": sentiment,
        "score": round(score, 4),
        "confidence": min(1.0, round(confidence, 2)),
        "key_phrases": found_phrases[:5],
        "impact_level": impact_level,
    }
