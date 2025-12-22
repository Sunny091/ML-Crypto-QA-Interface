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


# ===== Transformer Model =====

# Transformer feature set (60 days of OHLCV data)
TRANSFORMER_FEATURES = ["open", "high", "low", "close", "volume"]
TRANSFORMER_SEQUENCE_LENGTH = 60


class TransformerPredictor:
    """
    Transformer-based price predictor using attention mechanism.
    Falls back to statistical method if PyTorch is not available.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the model file. If None, uses config default.
        """
        if model_path is None:
            model_path = env.model.transformer_model_path

        self.model_path = model_path
        self.model: Any = None
        self.has_torch = False
        self.scaler: Any = None
        self.features = TRANSFORMER_FEATURES
        self.seq_length = TRANSFORMER_SEQUENCE_LENGTH

        # Check if PyTorch is available
        try:
            import torch
            self.has_torch = True
            print("[Model] PyTorch available, using Transformer model")
        except ImportError:
            print("[Model] PyTorch not available, using statistical fallback")

    def load(self) -> None:
        """Load the model from disk."""
        if not self.has_torch:
            print("[Model] No PyTorch, using statistical fallback (no model file needed)")
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Transformer model not found: {self.model_path}. "
                "Run 'python train_transformer.py' to create it."
            )

        import torch

        # Load model and scaler
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model = checkpoint['model']
        self.scaler = checkpoint.get('scaler')
        self.model.eval()
        print(f"[Model] Loaded Transformer model from {self.model_path}")

    def predict_statistical(self, df: pd.DataFrame) -> tuple[str, float, dict]:
        """
        Statistical fallback prediction using momentum and RSI.

        Args:
            df: DataFrame with OHLCV data (at least 60 rows)

        Returns:
            Tuple of (prediction, prob_up, attention_summary)
        """
        close = df['close'].values

        # Calculate momentum indicators
        returns = np.diff(close) / close[:-1]
        momentum_5d = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        momentum_10d = np.mean(returns[-10:]) if len(returns) >= 10 else 0

        # Calculate RSI
        gains = np.maximum(returns, 0)
        losses = np.abs(np.minimum(returns, 0))
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.01
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.01
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Calculate volatility
        volatility = np.std(returns[-7:]) if len(returns) >= 7 else 0.02

        # Combined score
        momentum_score = (momentum_5d * 0.6 + momentum_10d * 0.4) * 100
        rsi_score = (rsi - 50) / 50  # Normalize to [-1, 1]

        # Final probability (sigmoid-like)
        combined = momentum_score * 0.5 + rsi_score * 0.3
        prob_up = 1 / (1 + np.exp(-combined * 2))
        prob_up = float(np.clip(prob_up, 0.3, 0.7))  # Limit confidence

        prediction = "UP" if prob_up > 0.5 else "DOWN"

        attention_summary = {
            "method": "statistical_fallback",
            "momentum_5d": round(momentum_5d * 100, 4),
            "momentum_10d": round(momentum_10d * 100, 4),
            "rsi_14": round(rsi, 2),
            "volatility_7d": round(volatility * 100, 4),
        }

        return prediction, prob_up, attention_summary

    def predict(self, df: pd.DataFrame) -> tuple[str, float, dict]:
        """
        Make a prediction using Transformer or statistical fallback.

        Args:
            df: DataFrame with OHLCV data (at least 60 rows)

        Returns:
            Tuple of (prediction, prob_up, attention_summary)
        """
        # Ensure we have enough data
        if len(df) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} rows of data")

        # Use last seq_length rows
        df = df.tail(self.seq_length).copy()

        # Use statistical fallback if no PyTorch
        if not self.has_torch or self.model is None:
            return self.predict_statistical(df)

        import torch

        # Prepare data
        X = df[self.features].values

        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # (1, seq_len, features)

        # Predict
        with torch.no_grad():
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1)
            prob_up = float(probs[0, 1].item())

        prediction = "UP" if prob_up > 0.5 else "DOWN"

        # Get attention weights if available
        attention_summary = {
            "method": "transformer",
            "sequence_length": self.seq_length,
            "features": self.features,
        }

        # Add attention highlights if model supports it
        if hasattr(self.model, 'get_attention_weights'):
            weights = self.model.get_attention_weights()
            attention_summary["attention_peaks"] = weights.tolist()[-5:]

        return prediction, prob_up, attention_summary


# Global transformer instance
_transformer_predictor: TransformerPredictor | None = None


def get_transformer_predictor() -> TransformerPredictor:
    """
    Get the global Transformer predictor instance, loading if necessary.
    """
    global _transformer_predictor

    if _transformer_predictor is None:
        _transformer_predictor = TransformerPredictor()
        try:
            _transformer_predictor.load()
        except FileNotFoundError:
            print("[Model] Transformer model not found, using statistical fallback")

    return _transformer_predictor
