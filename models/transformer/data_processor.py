"""
Data Processor for Transformer Model

處理 Kaggle 加密貨幣歷史價格資料集
建立訓練/驗證/測試資料集
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class CryptoDataset(Dataset):
    """加密貨幣時間序列資料集"""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 30
    ):
        """
        Args:
            data: 特徵資料 [samples, features]
            labels: 標籤 [samples]
            seq_len: 序列長度
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 取 seq_len 天的資料作為輸入
        x = self.data[idx:idx + self.seq_len]
        # 取下一天的標籤
        y = self.labels[idx + self.seq_len]
        return x, y


class DataProcessor:
    """資料處理器"""

    # 技術指標特徵
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'return_1d', 'return_5d', 'rsi_14',
        'volatility_7d', 'ma_ratio'
    ]

    def __init__(
        self,
        data_dir: Path = None,
        seq_len: int = 30,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        self.data_dir = data_dir or PROJECT_ROOT / "data" / "raw"
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scaler_params = {}

    # 幣種符號對應的檔案名稱（Kaggle dataset 格式）
    COIN_FILES = {
        "BTC": "bitcoin.csv",
        "ETH": "ethereum.csv",
        "SOL": "solana.csv",
        "XRP": "XRP.csv",
        "BNB": "BNB.csv",
        "ADA": "cardano.csv",
        "DOGE": "dogecoin.csv",
        "Bitcoin": "bitcoin.csv",
        "Ethereum": "ethereum.csv",
        "Solana": "solana.csv",
    }

    def load_kaggle_data(self, symbol: str = "BTC") -> pd.DataFrame:
        """
        載入 Kaggle 資料集

        Dataset: Top 10 Cryptocurrencies Historical Dataset
        欄位: Date, Close, Open, High, Low, Vol.

        Args:
            symbol: 幣種符號 (BTC, ETH, SOL) 或名稱 (Bitcoin, Ethereum)

        Returns:
            DataFrame with OHLCV data
        """
        # 找到對應的檔案名稱
        filename = self.COIN_FILES.get(symbol)
        if not filename:
            print(f"[DataProcessor] 不支援的幣種: {symbol}，使用模擬資料")
            return self._generate_mock_data(symbol)

        file_path = self.data_dir / filename

        if not file_path.exists():
            print(f"[DataProcessor] {file_path} 不存在")
            print(f"[DataProcessor] 請先執行: python -m data.download_kaggle")
            raise FileNotFoundError(f"請先下載 Kaggle 資料集: {file_path}")

        print(f"[DataProcessor] 載入 {file_path}")
        df = pd.read_csv(file_path)

        # Kaggle 資料集格式: Date, Open, High, Low, Close, Volume, Currency
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 處理價格欄位 (移除逗號)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

        # 處理成交量欄位
        if 'Volume' in df.columns:
            df['volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        else:
            df['volume'] = 0

        # 重命名欄位
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
        })

        # 移除 NaN 價格
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        print(f"[DataProcessor] 載入 {len(df)} 筆資料 ({df['date'].min()} ~ {df['date'].max()})")

        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

    def _generate_mock_data(self, symbol: str, days: int = 1500) -> pd.DataFrame:
        """生成模擬資料（當 Kaggle 資料不可用時）"""
        print(f"[DataProcessor] 生成 {days} 天的模擬資料")

        np.random.seed(42)

        base_prices = {"Bitcoin": 30000, "Ethereum": 2000, "Solana": 100}
        base_price = base_prices.get(symbol, 100)

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = [base_price]

        for _ in range(days - 1):
            change = np.random.normal(0.0005, 0.025)
            prices.append(prices[-1] * (1 + change))

        prices = np.array(prices)
        highs = prices * (1 + np.abs(np.random.normal(0, 0.02, days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.02, days)))
        opens = lows + (highs - lows) * np.random.random(days)
        volumes = np.random.lognormal(20, 1, days)

        return pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標特徵（增強版）"""
        df = df.copy()

        # ===== 價格變化特徵 =====
        # 報酬率（多個時間尺度）
        df['return_1d'] = df['close'].pct_change(1)
        df['return_3d'] = df['close'].pct_change(3)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # 對數報酬率（更穩定）
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))

        # ===== RSI 指標（多個週期）=====
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # ===== 波動率（多個時間窗口）=====
        df['volatility_5d'] = df['return_1d'].rolling(window=5).std()
        df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
        df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

        # ===== 移動平均線 =====
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()

        # MA 比率
        df['ma_ratio_5'] = df['close'] / (df['ma_5'] + 1e-10)
        df['ma_ratio_20'] = df['close'] / (df['ma_20'] + 1e-10)
        df['ma_ratio_50'] = df['close'] / (df['ma_50'] + 1e-10)

        # MA 交叉信號
        df['ma_cross_5_20'] = (df['ma_5'] - df['ma_20']) / (df['ma_20'] + 1e-10)

        # ===== MACD =====
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ===== Bollinger Bands =====
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # ===== 價格位置特徵 =====
        df['high_low_ratio'] = (df['high'] - df['low']) / (df['low'] + 1e-10)
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # ===== 成交量特徵 =====
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)

        # ===== 動量指標 =====
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)

        # ===== 標籤：未來 5 天趨勢（更長期更穩定）=====
        # 使用未來5天的平均價格 vs 當前價格
        future_avg = df['close'].shift(-1).rolling(window=5, min_periods=3).mean().shift(-4)
        df['target'] = (future_avg > df['close'] * 1.001).astype(int)  # 需要超過0.1%才算漲

        return df

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """正規化資料"""
        if fit:
            self.scaler_params['mean'] = np.nanmean(data, axis=0)
            self.scaler_params['std'] = np.nanstd(data, axis=0) + 1e-10

        return (data - self.scaler_params['mean']) / self.scaler_params['std']

    # 特徵欄位（增強版 - 20個特徵）
    FEATURE_COLS = [
        # 報酬率特徵
        'return_1d', 'return_3d', 'return_5d', 'return_10d', 'log_return_1d',
        # RSI 指標
        'rsi_7', 'rsi_14', 'rsi_21',
        # 波動率
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        # MA 相關
        'ma_ratio_5', 'ma_ratio_20', 'ma_cross_5_20',
        # MACD
        'macd_hist',
        # Bollinger Bands
        'bb_width', 'bb_position',
        # 其他
        'high_low_ratio', 'close_position', 'volume_ratio'
    ]

    def prepare_data(
        self,
        symbol: str = "BTC",
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
        """
        準備訓練、驗證、測試資料

        Returns:
            train_loader, val_loader, test_loader, info_dict
        """
        # 載入資料
        df = self.load_kaggle_data(symbol)
        print(f"[DataProcessor] 原始資料: {len(df)} 筆")

        # 添加特徵
        df = self.add_features(df)

        # 移除 NaN
        df = df.dropna().reset_index(drop=True)
        print(f"[DataProcessor] 處理後資料: {len(df)} 筆")

        # 檢查類別平衡
        up_ratio = df['target'].mean()
        print(f"[DataProcessor] 標籤分布: UP={up_ratio:.2%}, DOWN={1-up_ratio:.2%}")

        # 提取特徵和標籤
        features = df[self.FEATURE_COLS].values
        labels = df['target'].values

        # 分割資料集（時間序列不能隨機分割）
        n = len(features)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_features = features[:train_end]
        val_features = features[train_end:val_end]
        test_features = features[val_end:]

        train_labels = labels[:train_end]
        val_labels = labels[train_end:val_end]
        test_labels = labels[val_end:]

        # 正規化（只用訓練資料計算參數）
        train_features = self.normalize(train_features, fit=True)
        val_features = self.normalize(val_features, fit=False)
        test_features = self.normalize(test_features, fit=False)

        # 處理異常值（clip 到 [-5, 5] 標準差範圍）
        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)
        test_features = np.clip(test_features, -5, 5)

        # 建立 Dataset
        train_dataset = CryptoDataset(train_features, train_labels, self.seq_len)
        val_dataset = CryptoDataset(val_features, val_labels, self.seq_len)
        test_dataset = CryptoDataset(test_features, test_labels, self.seq_len)

        # 建立 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        info = {
            "symbol": symbol,
            "total_samples": n,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "num_features": len(self.FEATURE_COLS),
            "features": self.FEATURE_COLS,
            "seq_len": self.seq_len,
            "date_range": f"{df['date'].min()} ~ {df['date'].max()}",
            "up_ratio": f"{up_ratio:.2%}"
        }

        print(f"[DataProcessor] 訓練: {info['train_samples']}, 驗證: {info['val_samples']}, 測試: {info['test_samples']}")
        print(f"[DataProcessor] 特徵數: {info['num_features']}")

        return train_loader, val_loader, test_loader, info


# 測試
if __name__ == "__main__":
    processor = DataProcessor()
    train_loader, val_loader, test_loader, info = processor.prepare_data("Bitcoin")

    print("\n資料集資訊:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 檢查一個 batch
    for x, y in train_loader:
        print(f"\nBatch shape: x={x.shape}, y={y.shape}")
        print(f"Labels distribution: {y.sum().item()}/{len(y)} 漲")
        break
