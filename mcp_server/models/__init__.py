"""
Models module for loading and using trained models.
Supports price-only (Phase 2) and price+text (Phase 4) models.
"""

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import env
from features import PRICE_ONLY_FEATURES, PRICE_TEXT_FEATURES


class PriceOnlyPredictor:
    """
    Wrapper for the price-only LightGBM model.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the model file. If None, uses config default.
        """
        if model_path is None:
            model_path = env.model.price_only_model_path

        self.model_path = model_path
        self.model: Any = None
        self.features = PRICE_ONLY_FEATURES

    def load(self) -> None:
        """Load the model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Run 'python train_model.py' to create it."
            )

        self.model = joblib.load(self.model_path)
        print(f"[Model] Loaded model from {self.model_path}")

    def predict(self, features: dict[str, float]) -> tuple[str, float]:
        """
        Make a prediction given feature values.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Tuple of (prediction: "UP" or "DOWN", prob_up: float)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create feature DataFrame with proper column names
        X = pd.DataFrame([[features[f] for f in self.features]], columns=self.features)

        # Get probability of UP (class 1)
        prob = self.model.predict_proba(X)[0]
        prob_up = float(prob[1])

        prediction = "UP" if prob_up > 0.5 else "DOWN"

        return prediction, prob_up


# ===== Phase 4: Price + Text Model =====

class PriceTextPredictor:
    """
    Wrapper for the price+text LightGBM model.
    Uses both price features and news sentiment.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the model file. If None, uses config default.
        """
        if model_path is None:
            model_path = env.model.price_text_model_path

        self.model_path = model_path
        self.model: Any = None
        self.features = PRICE_TEXT_FEATURES

    def load(self) -> None:
        """Load the model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Run 'python train_model.py --with-text' to create it."
            )

        self.model = joblib.load(self.model_path)
        print(f"[Model] Loaded price+text model from {self.model_path}")

    def predict(self, features: dict[str, float]) -> tuple[str, float]:
        """
        Make a prediction given feature values.

        Args:
            features: Dictionary of feature name -> value
                      Must include: return_1d, rsi_14, volatility_7d,
                                   sentiment_score, news_present

        Returns:
            Tuple of (prediction: "UP" or "DOWN", prob_up: float)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create feature DataFrame with proper column names
        X = pd.DataFrame([[features[f] for f in self.features]], columns=self.features)

        # Get probability of UP (class 1)
        prob = self.model.predict_proba(X)[0]
        prob_up = float(prob[1])

        prediction = "UP" if prob_up > 0.5 else "DOWN"

        return prediction, prob_up


# Global model instances (loaded once at startup)
_predictor: PriceOnlyPredictor | None = None
_text_predictor: PriceTextPredictor | None = None


def get_predictor() -> PriceOnlyPredictor:
    """
    Get the global price-only predictor instance, loading if necessary.
    """
    global _predictor

    if _predictor is None:
        _predictor = PriceOnlyPredictor()
        _predictor.load()

    return _predictor


def get_text_predictor() -> PriceTextPredictor:
    """
    Get the global price+text predictor instance, loading if necessary.
    """
    global _text_predictor

    if _text_predictor is None:
        _text_predictor = PriceTextPredictor()
        _text_predictor.load()

    return _text_predictor
