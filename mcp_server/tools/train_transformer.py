#!/usr/bin/env python3
"""
Transformer Model Training Script

Trains a Transformer-based model for cryptocurrency price prediction.
Uses 60 days of OHLCV data to predict next-day price movement.

Usage:
    python train_transformer.py [--symbol BTC] [--epochs 50] [--mock]

Requirements:
    - PyTorch >= 2.0.0
    - pandas, numpy, scikit-learn
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import env


# ===== Transformer Model Definition =====

def create_transformer_model(input_dim: int = 5, d_model: int = 64, nhead: int = 4,
                             num_layers: int = 2, dropout: float = 0.1):
    """
    Create the Transformer model for price prediction.

    Architecture:
        Input (seq_len × 5) -> Linear Projection (64)
        -> Positional Encoding
        -> Transformer Encoder (2 layers, 4 heads)
        -> Global Average Pooling
        -> Classifier (2 classes: UP/DOWN)
    """
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class CryptoPriceTransformer(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)
            )

            self._attention_weights = None

        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)

            # Global average pooling
            x = x.mean(dim=1)

            # Classification
            return self.classifier(x)

        def get_attention_weights(self):
            return self._attention_weights

    return CryptoPriceTransformer(input_dim, d_model, nhead, num_layers, dropout)


# ===== Data Preparation =====

def generate_mock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Generate mock OHLCV data for training.
    Uses realistic price movements with trends and volatility.
    """
    np.random.seed(42)  # Reproducible

    base_prices = {"BTC": 45000, "ETH": 2500, "SOL": 100}
    base_price = base_prices.get(symbol, 45000)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    prices = [base_price]

    for i in range(1, days):
        # Add trend, mean reversion, and noise
        trend = 0.0002  # Slight upward trend
        mean_rev = -0.001 * (prices[-1] - base_price) / base_price
        noise = np.random.randn() * 0.02  # 2% daily volatility

        change = trend + mean_rev + noise
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, base_price * 0.5))  # Floor at 50%

    # Generate OHLCV from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        volatility = 0.01 + np.random.rand() * 0.02
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close

        volume = base_price * 1000 * (0.5 + np.random.rand())

        data.append({
            "date": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)


def prepare_sequences(df: pd.DataFrame, seq_length: int = 60) -> tuple:
    """
    Prepare sequence data for training.

    Args:
        df: DataFrame with OHLCV columns
        seq_length: Number of days in each sequence

    Returns:
        X: (samples, seq_length, 5) array
        y: (samples,) array of labels (0=DOWN, 1=UP)
    """
    features = ["open", "high", "low", "close", "volume"]
    data = df[features].values

    # Create sequences and labels
    X, y = [], []

    for i in range(len(data) - seq_length - 1):
        # Input: seq_length days of OHLCV
        X.append(data[i:i+seq_length])

        # Label: 1 if next day close > today close, else 0
        today_close = data[i + seq_length - 1, 3]  # close is index 3
        next_close = data[i + seq_length, 3]
        y.append(1 if next_close > today_close else 0)

    return np.array(X), np.array(y)


# ===== Training =====

def train_model(X_train, y_train, X_val, y_val, epochs: int = 50, lr: float = 0.001):
    """
    Train the Transformer model.

    Returns:
        Trained model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    model = create_transformer_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Transformer model for crypto prediction")
    parser.add_argument("--symbol", type=str, default="BTC", help="Cryptocurrency symbol")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--mock", action="store_true", help="Use mock data (for testing)")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    args = parser.parse_args()

    print("=" * 50)
    print("Transformer Model Training")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Epochs: {args.epochs}")
    print(f"Data source: {'mock' if args.mock else 'API'}")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    # Get data
    print("\n[1/4] Loading data...")
    if args.mock:
        df = generate_mock_data(args.symbol, days=400)
    else:
        # Try to get real data from API
        try:
            from api import get_price_history
            history = get_price_history(args.symbol, interval="d1", days=365)
            df = pd.DataFrame(history["data"])
            df["date"] = pd.to_datetime(df["timestamp"])
            # For API data, we only have price, need to generate OHLCV
            df["open"] = df["price_usd"]
            df["high"] = df["price_usd"] * 1.01
            df["low"] = df["price_usd"] * 0.99
            df["close"] = df["price_usd"]
            df["volume"] = 1000000
        except Exception as e:
            print(f"API failed ({e}), using mock data")
            df = generate_mock_data(args.symbol, days=400)

    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Prepare sequences
    print("\n[2/4] Preparing sequences...")
    X, y = prepare_sequences(df, seq_length=60)
    print(f"  Sequences: {X.shape[0]}")
    print(f"  Class distribution: UP={y.sum()}, DOWN={len(y)-y.sum()}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = X.copy()
    for i in range(X.shape[0]):
        X_scaled[i] = scaler.fit_transform(X[i])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # Train model
    print("\n[3/4] Training model...")
    model = train_model(X_train, y_train, X_val, y_val, epochs=args.epochs)

    # Save model
    print("\n[4/4] Saving model...")
    output_path = args.output or env.model.transformer_model_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model,
        "scaler": scaler,
        "symbol": args.symbol,
        "seq_length": 60,
        "created_at": datetime.now().isoformat(),
    }
    torch.save(checkpoint, output_path)
    print(f"  Model saved to: {output_path}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
