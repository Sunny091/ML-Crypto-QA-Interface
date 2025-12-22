"""
Transformer 模型預測器

用於 MCP Tool 整合，提供預測功能
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TransformerPredictor:
    """Transformer 模型預測器"""

    def __init__(self, model_path: Path = None, scaler_path: Path = None):
        self.model_path = model_path or PROJECT_ROOT / "models" / "artifacts" / "transformer.pt"
        self.scaler_path = scaler_path or PROJECT_ROOT / "models" / "artifacts" / "transformer_scaler.json"
        self.model = None
        self.scaler_params = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> bool:
        """載入模型"""
        if not self.model_path.exists():
            print(f"[TransformerPredictor] 模型檔案不存在: {self.model_path}")
            return False

        if not self.scaler_path.exists():
            print(f"[TransformerPredictor] Scaler 檔案不存在: {self.scaler_path}")
            return False

        try:
            # 載入模型
            from models.transformer.model import create_model

            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.config = checkpoint['config']

            self.model = create_model(self.config['model']).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # 載入 scaler
            with open(self.scaler_path, 'r') as f:
                self.scaler_params = json.load(f)

            print(f"[TransformerPredictor] 模型載入成功 (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")
            return True

        except Exception as e:
            print(f"[TransformerPredictor] 載入失敗: {e}")
            return False

    def _prepare_features(self, price_data: list) -> np.ndarray:
        """
        準備特徵資料

        Args:
            price_data: 價格資料列表 [{date, open, high, low, close, volume}, ...]

        Returns:
            特徵陣列 [seq_len, num_features]
        """
        import pandas as pd

        df = pd.DataFrame(price_data)

        # 計算技術指標
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)

        # RSI 14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 波動率
        df['volatility_7d'] = df['return_1d'].rolling(window=7).std()

        # MA ratio
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_ratio'] = df['close'] / (df['ma_20'] + 1e-10)

        # 取最後 seq_len 筆
        seq_len = self.config['model']['seq_len']
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                       'return_1d', 'return_5d', 'rsi_14', 'volatility_7d', 'ma_ratio']

        features = df[feature_cols].values[-seq_len:]

        # 填補 NaN
        features = np.nan_to_num(features, nan=0.0)

        # 正規化
        mean = np.array(self.scaler_params['mean'])
        std = np.array(self.scaler_params['std'])
        features = (features - mean) / std

        return features

    def predict(self, price_data: list) -> Dict:
        """
        預測明天漲跌

        Args:
            price_data: 至少 50 天的價格資料

        Returns:
            預測結果 dict
        """
        if self.model is None:
            if not self.load():
                return {"error": "模型載入失敗"}

        try:
            # 準備特徵
            features = self._prepare_features(price_data)

            # 轉換為 tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # [1, seq_len, features]

            # 預測
            with torch.no_grad():
                proba = self.model.predict_proba(x)[0]

            down_prob = proba[0].item()
            up_prob = proba[1].item()

            prediction = "UP" if up_prob > down_prob else "DOWN"
            confidence = max(up_prob, down_prob)

            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "UP": round(up_prob, 4),
                    "DOWN": round(down_prob, 4)
                },
                "model": "Transformer",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_predictor: Optional[TransformerPredictor] = None


def get_transformer_predictor() -> TransformerPredictor:
    """取得 Transformer 預測器單例"""
    global _predictor
    if _predictor is None:
        _predictor = TransformerPredictor()
    return _predictor


def predict_with_transformer(symbol: str) -> Dict:
    """
    使用 Transformer 模型預測

    Args:
        symbol: 幣種符號 (BTC, ETH, SOL)

    Returns:
        預測結果
    """
    from data.scrapers.coincap_client import get_price_history

    # 取得歷史資料 (至少需要 50 天來計算指標)
    history = get_price_history(symbol, days=60)

    if not history.get("data"):
        return {"error": "無法取得價格資料"}

    # 轉換資料格式
    price_data = []
    for point in history["data"]:
        # 模擬 OHLCV（因為 API 只返回 close price）
        close = point["price_usd"]
        price_data.append({
            "date": point["timestamp"],
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e9  # 模擬成交量
        })

    predictor = get_transformer_predictor()
    result = predictor.predict(price_data)
    result["symbol"] = symbol
    result["data_source"] = history["source"]

    return result


# 測試
if __name__ == "__main__":
    result = predict_with_transformer("BTC")
    print(json.dumps(result, indent=2, ensure_ascii=False))
