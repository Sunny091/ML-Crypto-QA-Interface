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


def _is_chinese_text(text: str) -> bool:
    """Check if text contains significant Chinese characters"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars / max(len(text), 1) > 0.3


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

    # Use mock sentiment for Chinese text (FinBERT is trained on English)
    if _is_chinese_text(text):
        return _mock_sentiment(text)

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


# ===== Enhanced Sentiment Analysis =====

# High-impact keywords that strongly influence price
HIGH_IMPACT_KEYWORDS = {
    "positive": [
        "etf approved", "etf approval", "institutional adoption", "bitcoin etf",
        "spot etf", "record high", "all-time high", "ath", "breakthrough",
        "批准", "etf通過", "創新高", "歷史新高", "機構入場",
    ],
    "negative": [
        "sec lawsuit", "exchange hack", "major exploit", "bankruptcy",
        "exchange collapse", "ponzi scheme", "rug pull", "exit scam",
        "訴訟", "駭客攻擊", "交易所倒閉", "破產", "詐騙",
    ],
}

# Crypto-specific keywords for extraction
CRYPTO_KEYWORDS = {
    "positive": [
        "surge", "soar", "rally", "bullish", "gain", "rise", "jump", "moon",
        "breakthrough", "adoption", "institutional", "etf", "approval",
        "growth", "profit", "milestone", "record", "high", "buy", "accumulate",
        "whale", "halving", "upgrade", "mainnet", "launch",
        "上漲", "突破", "利好", "看漲", "成長", "創新高", "買入", "減半", "升級",
    ],
    "negative": [
        "crash", "plunge", "bearish", "drop", "fall", "decline", "sell",
        "ban", "regulation", "hack", "fraud", "scam", "collapse", "risk",
        "loss", "dump", "fear", "uncertainty", "lawsuit", "sec", "fud",
        "liquidation", "capitulation", "panic",
        "下跌", "暴跌", "利空", "看跌", "禁止", "監管", "風險", "清算", "恐慌",
    ],
    "neutral": [
        "bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft",
        "trading", "market", "price", "volume", "analysis",
        "比特幣", "以太坊", "加密貨幣", "區塊鏈",
    ],
}


def extract_key_phrases(text: str) -> list[str]:
    """
    Extract key phrases and keywords from news text.

    Args:
        text: News article text

    Returns:
        List of extracted key phrases (max 5)
    """
    text_lower = text.lower()
    found_phrases = []

    # Check high-impact phrases first
    for category in ["positive", "negative"]:
        for phrase in HIGH_IMPACT_KEYWORDS[category]:
            if phrase in text_lower:
                found_phrases.append(phrase)

    # Then check regular keywords
    all_keywords = (
        CRYPTO_KEYWORDS["positive"]
        + CRYPTO_KEYWORDS["negative"]
        + CRYPTO_KEYWORDS["neutral"]
    )

    for keyword in all_keywords:
        if keyword in text_lower and keyword not in found_phrases:
            found_phrases.append(keyword)

    # Deduplicate and limit to 5
    return list(dict.fromkeys(found_phrases))[:5]


def assess_impact_level(text: str, sentiment_score: float) -> Literal["HIGH", "MEDIUM", "LOW"]:
    """
    Assess the potential price impact level of news.

    Args:
        text: News article text
        sentiment_score: Pre-calculated sentiment score

    Returns:
        Impact level: HIGH, MEDIUM, or LOW
    """
    text_lower = text.lower()

    # Check for high-impact keywords
    for category in ["positive", "negative"]:
        for phrase in HIGH_IMPACT_KEYWORDS[category]:
            if phrase in text_lower:
                return "HIGH"

    # Strong sentiment = higher impact
    abs_sentiment = abs(sentiment_score)
    if abs_sentiment > 0.7:
        return "HIGH"
    elif abs_sentiment > 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_confidence(text: str, sentiment_score: float) -> float:
    """
    Calculate confidence in the sentiment analysis.

    Args:
        text: News article text
        sentiment_score: Pre-calculated sentiment score

    Returns:
        Confidence score in range [0.0, 1.0]
    """
    text_lower = text.lower()

    # Start with base confidence based on sentiment strength
    abs_sentiment = abs(sentiment_score)
    confidence = 0.5 + (abs_sentiment * 0.3)  # Range: [0.5, 0.8]

    # Boost confidence for high-impact keywords
    for category in ["positive", "negative"]:
        for phrase in HIGH_IMPACT_KEYWORDS[category]:
            if phrase in text_lower:
                confidence += 0.1
                break

    # Boost for longer text (more context)
    if len(text) > 500:
        confidence += 0.05
    if len(text) > 1000:
        confidence += 0.05

    return min(1.0, round(confidence, 2))


def analyze_news_full(text: str) -> dict:
    """
    Full news sentiment analysis with all details.

    Args:
        text: News article text

    Returns:
        Dictionary with:
        - sentiment: BULLISH/BEARISH/NEUTRAL
        - score: float in [-1, 1]
        - confidence: float in [0, 1]
        - key_phrases: list of extracted phrases
        - impact_level: HIGH/MEDIUM/LOW
    """
    if not text or not text.strip():
        return {
            "sentiment": "NEUTRAL",
            "score": 0.0,
            "confidence": 0.0,
            "key_phrases": [],
            "impact_level": "LOW",
        }

    # Get base sentiment score
    score = analyze_sentiment(text)

    # Determine sentiment label
    if score > 0.1:
        sentiment = "BULLISH"
    elif score < -0.1:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    # Extract key phrases
    key_phrases = extract_key_phrases(text)

    # Assess impact
    impact_level = assess_impact_level(text, score)

    # Calculate confidence
    confidence = calculate_confidence(text, score)

    return {
        "sentiment": sentiment,
        "score": round(score, 4),
        "confidence": confidence,
        "key_phrases": key_phrases,
        "impact_level": impact_level,
    }
