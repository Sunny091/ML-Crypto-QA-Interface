"""
LSTM Model for Crypto Price Prediction

架構：
- Bidirectional LSTM layers
- Dropout for regularization
- Attention mechanism for better sequence modeling
- 輸出：漲跌分類 (UP/DOWN)
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Attention mechanism for LSTM outputs"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]
        Returns:
            context: [batch, hidden_dim]
        """
        # 計算注意力權重
        attn_weights = self.attention(lstm_output)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)

        # 加權求和
        context = torch.sum(lstm_output * attn_weights, dim=1)  # [batch, hidden_dim]
        return context


class CryptoLSTM(nn.Module):
    """
    LSTM 模型用於加密貨幣價格預測

    特點：
    - 雙向 LSTM 捕捉前後文信息
    - 多層 LSTM 學習複雜模式
    - Attention 機制聚焦重要時間步
    - Dropout 防止過擬合
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 輸入投影層
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Layer Normalization
        lstm_output_dim = hidden_dim * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Attention 層
        self.attention = Attention(lstm_output_dim)

        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化權重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, input_dim]

        Returns:
            logits: Tensor, shape [batch_size, num_classes]
        """
        batch_size = x.size(0)

        # 輸入投影
        x = self.input_projection(x)  # [batch, seq, hidden_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq, hidden_dim * num_directions]

        # Layer Norm
        lstm_out = self.layer_norm(lstm_out)

        # Attention pooling
        context = self.attention(lstm_out)  # [batch, hidden_dim * num_directions]

        # 分類
        logits = self.classifier(context)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """預測機率"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def create_model(config: dict = None) -> CryptoLSTM:
    """建立模型"""
    default_config = {
        "input_dim": 20,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
        "num_classes": 2
    }

    if config:
        default_config.update(config)

    return CryptoLSTM(**default_config)


# 測試
if __name__ == "__main__":
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 測試輸入
    batch_size = 4
    seq_len = 30
    input_dim = 20
    x = torch.randn(batch_size, seq_len, input_dim)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    proba = model.predict_proba(x)
    print(f"Probabilities: {proba[0]}")
