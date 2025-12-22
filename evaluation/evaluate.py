"""
Model Evaluation Module - 模型評估與成果展示

Provides comprehensive model evaluation including:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC Score
- Feature Importance Analysis
- Cross-validation

Usage:
    python -m evaluation.evaluate
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import env


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # True/False Positive/Negative
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negative"] = int(tn)
        metrics["false_positive"] = int(fp)
        metrics["false_negative"] = int(fn)
        metrics["true_positive"] = int(tp)

    # ROC-AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str] = None,
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names for importance analysis

    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = calculate_metrics(y_test.values, y_pred, y_prob)

    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")

    if metrics.get('roc_auc'):
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Feature importance
    if hasattr(model, 'feature_importances_') and feature_names:
        print("\nFeature Importance:")
        importance = dict(zip(feature_names, model.feature_importances_))
        metrics["feature_importance"] = importance

        for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.4f}")

    return metrics


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict:
    """
    Perform cross-validation on a model.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of cross-validation folds

    Returns:
        Dictionary with cross-validation results
    """
    print(f"\n{cv}-Fold Cross Validation:")

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    results = {
        "cv_scores": scores.tolist(),
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
    }

    print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean:   {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")

    return results


def generate_evaluation_report(
    model_name: str,
    metrics: dict,
    output_dir: Path = None,
) -> Path:
    """
    Generate a JSON evaluation report.

    Args:
        model_name: Name of the model
        metrics: Evaluation metrics dictionary
        output_dir: Directory to save the report

    Returns:
        Path to the generated report
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "evaluation" / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    report = convert_types(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{model_name}_evaluation_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    return report_path


def run_full_evaluation():
    """Run full evaluation on all trained models."""
    from data.scrapers.mock_generator import generate_mock_price_data
    from etl.transform import prepare_training_data, PRICE_ONLY_FEATURES
    import joblib

    print("=" * 60)
    print("Full Model Evaluation")
    print("=" * 60)

    # Prepare test data
    print("\n[Data] Preparing evaluation data...")
    all_X = []
    all_y = []

    for symbol in ["BTC", "ETH", "SOL"]:
        df = generate_mock_price_data(symbol, "2024-01-01", "2024-06-30")
        X, y = prepare_training_data(df)
        all_X.append(X)
        all_y.append(y)

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    print(f"[Data] Total samples: {len(X)}")

    # Evaluate price-only model
    model_path = env.model.price_only_model_path
    if model_path.exists():
        print(f"\n[Model] Loading {model_path}")
        model = joblib.load(model_path)

        metrics = evaluate_model(model, X, y, PRICE_ONLY_FEATURES)
        cv_results = cross_validate_model(model, X, y)
        metrics.update(cv_results)

        generate_evaluation_report("price_only_lgbm", metrics)
    else:
        print(f"\n[Warning] Model not found: {model_path}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_evaluation()
