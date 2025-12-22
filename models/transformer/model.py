"""
Transformer Model for Crypto Price Prediction

架構：
- Positional Encoding
- Multi-Head Self-Attention
- Feed Forward Network
- 輸出：漲跌分類 (UP/DOWN)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """位置編碼層"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 計算位置編碼
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CryptoTransformer(nn.Module):
    """
    Transformer 模型用於加密貨幣價格預測（優化版）

    改進：
    - 更深的網路結構
    - Layer Normalization
    - 殘差連接
    - 多尺度特徵融合
    - Attention Pooling
    """

    def __init__(
        self,
        input_dim: int = 20,       # 輸入特徵數量（增加）
        d_model: int = 128,        # Transformer 隱藏維度（增加）
        nhead: int = 8,            # 注意力頭數（增加）
        num_layers: int = 4,       # Transformer 層數（增加）
        dim_feedforward: int = 256, # FFN 維度（增加）
        dropout: float = 0.2,      # Dropout（增加）
        seq_len: int = 60,         # 序列長度（增加）
        num_classes: int = 2       # 分類數量 (UP/DOWN)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # 輸入投影層（帶 LayerNorm）
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer Encoder（使用 batch_first=True）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Attention Pooling（取代只用最後一個時間步）
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # 分類頭（更深）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化權重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, input_dim]

        Returns:
            logits: Tensor, shape [batch_size, num_classes]
        """
        batch_size = x.size(0)

        # [batch, seq, input_dim] -> [batch, seq, d_model]
        x = self.input_projection(x)

        # 位置編碼（調整為 batch_first）
        # [batch, seq, d_model] -> [seq, batch, d_model] -> pos -> [batch, seq, d_model]
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        # Transformer Encoder (batch_first=True)
        x = self.transformer_encoder(x)  # [batch, seq, d_model]

        # Attention Pooling: 加權平均所有時間步
        attn_weights = self.attention_pool(x)  # [batch, seq, 1]
        x = torch.sum(x * attn_weights, dim=1)  # [batch, d_model]

        # 分類
        logits = self.classifier(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """預測機率"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def create_model(config: dict = None) -> CryptoTransformer:
    """建立模型"""
    default_config = {
        "input_dim": 10,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "seq_len": 30,
        "num_classes": 2
    }

    if config:
        default_config.update(config)

    return CryptoTransformer(**default_config)


# 測試
if __name__ == "__main__":
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 測試輸入
    batch_size = 4
    seq_len = 30
    input_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    proba = model.predict_proba(x)
    print(f"Probabilities: {proba[0]}")
