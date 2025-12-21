#!/usr/bin/env python3
"""
Training script for LightGBM models.

Supports:
- Price-only model (Phase 2)
- Price+text model (Phase 4)

Usage:
    python train_model.py              # Train price-only model
    python train_model.py --with-text  # Train price+text model
    python train_model.py --all        # Train both models
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import env
from data import generate_mock_price_data
from features import (
    prepare_training_data,
    prepare_training_data_with_text,
    PRICE_ONLY_FEATURES,
    PRICE_TEXT_FEATURES,
)


def train_price_only_model():
    """Train the price-only model (Phase 2)."""
    print("=" * 60)
    print("Training Price-Only LightGBM Model")
    print("=" * 60)

    # Collect training data from all symbols
    all_X = []
    all_y = []

    for symbol in ["BTC", "ETH", "SOL"]:
        print(f"\n[Data] Generating mock data for {symbol}...")
        df = generate_mock_price_data(symbol, "2022-01-01", "2024-06-30")
        X, y = prepare_training_data(df)

        print(f"  - Samples: {len(X)}")
        print(f"  - Features: {list(X.columns)}")
        print(f"  - Target distribution: UP={y.sum()}, DOWN={len(y) - y.sum()}")

        all_X.append(X)
        all_y.append(y)

    # Combine all data
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    print(f"\n[Data] Total samples: {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[Data] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    print("\n[Training] Training LightGBM classifier...")

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[Evaluation] Test Accuracy: {accuracy:.4f}")
    print("\n[Evaluation] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Feature importance
    print("\n[Model] Feature Importance:")
    for name, importance in zip(PRICE_ONLY_FEATURES, model.feature_importances_):
        print(f"  - {name}: {importance:.4f}")

    # Save model
    model_dir = env.model.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = env.model.price_only_model_path
    joblib.dump(model, model_path)

    print(f"\n[Save] Model saved to: {model_path}")
    print("=" * 60)


def train_price_text_model():
    """Train the price+text model (Phase 4)."""
    print("=" * 60)
    print("Training Price+Text LightGBM Model")
    print("=" * 60)

    # Collect training data from all symbols
    all_X = []
    all_y = []

    for symbol in ["BTC", "ETH", "SOL"]:
        print(f"\n[Data] Generating mock data for {symbol}...")
        df = generate_mock_price_data(symbol, "2022-01-01", "2024-06-30")
        X, y = prepare_training_data_with_text(df)

        print(f"  - Samples: {len(X)}")
        print(f"  - Features: {list(X.columns)}")
        print(f"  - Target distribution: UP={y.sum()}, DOWN={len(y) - y.sum()}")

        all_X.append(X)
        all_y.append(y)

    # Combine all data
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    print(f"\n[Data] Total samples: {len(X)}")

    # Show sentiment distribution
    print(f"\n[Data] Sentiment Stats:")
    print(f"  - Mean sentiment_score: {X['sentiment_score'].mean():.4f}")
    print(f"  - News present ratio: {X['news_present'].mean():.2%}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[Data] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    print("\n[Training] Training LightGBM classifier...")

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[Evaluation] Test Accuracy: {accuracy:.4f}")
    print("\n[Evaluation] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Feature importance
    print("\n[Model] Feature Importance:")
    for name, importance in zip(PRICE_TEXT_FEATURES, model.feature_importances_):
        print(f"  - {name}: {importance:.4f}")

    # Save model
    model_dir = env.model.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = env.model.price_text_model_path
    joblib.dump(model, model_path)

    print(f"\n[Save] Model saved to: {model_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM models")
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="Train price+text model (Phase 4)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train both price-only and price+text models"
    )

    args = parser.parse_args()

    if args.all:
        train_price_only_model()
        print("\n")
        train_price_text_model()
    elif args.with_text:
        train_price_text_model()
    else:
        train_price_only_model()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
