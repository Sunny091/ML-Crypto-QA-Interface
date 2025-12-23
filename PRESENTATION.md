# 加密貨幣價格預測系統
## 自動化從資料收集到模型訓練與部署

**機器學習專題報告**

---

## 1. 專題題目與動機

### 題目
**加密貨幣智慧問答與價格預測系統**
- 結合 NLP 自然語言處理與深度學習的端到端 MLOps 專案

### 動機
1. **市場需求**：加密貨幣市場波動劇烈，投資者需要即時且智慧的分析工具
2. **技術整合**：希望實踐完整的 MLOps 流程，從資料收集、清理、訓練到部署
3. **學習目標**：
   - 實作 ETL Pipeline 自動化資料處理
   - 訓練 Transformer 與 LSTM 深度學習模型
   - 使用 Flask 部署 RESTful API
   - 整合 LLM（Ollama）實現自然語言問答

---

## 2. 系統架構圖（ETL / Pipeline）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         系統架構總覽                                  │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   使用者      │───▶│   Flask API  │───▶│  LLM Agent   │
│  (Web UI)    │◀───│   (app.py)   │◀───│  (Ollama)    │
└──────────────┘    └──────────────┘    └──────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐    ┌──────────────┐
                    │  MCP Tools   │    │ Tool Calling │
                    │  (工具層)     │◀───│  (Function)  │
                    └──────────────┘    └──────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ 即時價格    │   │ 歷史價格    │   │ ML 預測    │
   │ CoinGecko  │   │ 走勢圖表    │   │Transformer │
   └────────────┘   └────────────┘   │   & LSTM   │
                                      └────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       ETL Pipeline                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐       │
│  │ Extract │────▶│Transform│────▶│  Load   │────▶│  Train  │       │
│  │ 資料爬蟲 │     │ 特徵工程 │     │ 資料集  │     │ 模型訓練 │       │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘       │
│       │               │               │               │             │
│       ▼               ▼               ▼               ▼             │
│  CoinGecko API   20個技術指標    Train/Val/Test   Transformer       │
│  Kaggle Dataset  RSI, MACD...   70/15/15 分割    LSTM Model         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 專案目錄結構
```
ML-Crypto-QA-Interface/
├── api/                    # Flask API 服務
│   ├── app.py             # 主應用程式
│   └── static/            # 前端 UI
├── data/                   # 資料層
│   ├── raw/               # 原始資料 (Kaggle)
│   └── scrapers/          # 資料爬蟲 (CoinGecko)
├── etl/                    # ETL Pipeline
│   ├── extract.py         # 資料擷取
│   └── transform.py       # 特徵工程
├── models/                 # 模型層
│   ├── transformer/       # Transformer 模型
│   │   ├── model.py      # 模型架構
│   │   ├── train.py      # 訓練腳本
│   │   └── predictor.py  # 預測器
│   ├── lstm/              # LSTM 模型
│   │   ├── model.py      # 模型架構
│   │   ├── train.py      # 訓練腳本
│   │   └── predictor.py  # 預測器
│   └── artifacts/         # 模型檔案
│       ├── transformer.pt # Transformer 權重
│       └── lstm.pt        # LSTM 權重
├── orchestrator/           # LLM Agent
│   └── agent.py           # ReAct 模式 Agent
├── mcp/                    # MCP 工具定義
│   └── tools.py           # 工具實作
└── evaluation/             # 評估結果
    └── results/           # 訓練報告
```

---

## 3. 資料面：DataOps

### 3.1 資料爬蟲（Extract）

**資料來源**
| 來源 | 用途 | 資料量 |
|------|------|--------|
| CoinGecko API | 即時價格、歷史走勢 | 即時 |
| Kaggle Dataset | 模型訓練 | 4,371 筆 (BTC) |

**CoinGecko API 整合** (`data/scrapers/coincap_client.py`)
```python
# 支援幣種
SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana"
}

# API 端點
- /simple/price          # 即時價格
- /coins/{id}/market_chart  # 歷史走勢
```

### 3.2 資料清理與特徵工程（Transform）

**20 個技術指標特徵**

| 類別 | 特徵名稱 | 說明 |
|------|----------|------|
| 報酬率 | return_1d, 3d, 5d, 10d | 多時間尺度報酬 |
| 報酬率 | log_return_1d | 對數報酬率 |
| RSI | rsi_7, rsi_14, rsi_21 | 相對強弱指標 |
| 波動率 | volatility_5d, 10d, 20d | 價格波動程度 |
| MA | ma_ratio_5, ma_ratio_20 | 移動平均比率 |
| MA | ma_cross_5_20 | 均線交叉信號 |
| MACD | macd_hist | MACD 柱狀圖 |
| BB | bb_width, bb_position | 布林通道指標 |
| 其他 | high_low_ratio | 最高最低比 |
| 其他 | close_position | 收盤價位置 |
| 其他 | volume_ratio | 成交量比率 |

**資料處理流程**
```python
# 1. 載入原始資料
df = load_kaggle_data("BTC")  # 4,371 筆

# 2. 計算技術指標
df = add_features(df)  # 20 個特徵

# 3. 正規化
features = normalize(features)  # Z-score

# 4. 分割資料集
Train: 70% (3,029 筆)
Val:   15% (626 筆)
Test:  15% (626 筆)
```

---

## 4. 模型面：MLOps

### 4.1 模型架構

本專案實作了兩種深度學習模型進行加密貨幣價格預測：

#### 4.1.1 Transformer 模型

```
┌─────────────────────────────────────────┐
│           CryptoTransformer             │
├─────────────────────────────────────────┤
│  Input: [batch, 30, 20]                 │
│         (序列長度=30, 特徵數=20)          │
├─────────────────────────────────────────┤
│  1. Input Projection                    │
│     Linear(20 → 64) + LayerNorm + GELU  │
├─────────────────────────────────────────┤
│  2. Positional Encoding                 │
│     位置編碼 (sin/cos)                   │
├─────────────────────────────────────────┤
│  3. Transformer Encoder (x2 layers)     │
│     - Multi-Head Attention (4 heads)    │
│     - Feed Forward (128 dim)            │
│     - Dropout (0.4)                     │
├─────────────────────────────────────────┤
│  4. Attention Pooling                   │
│     加權平均所有時間步                    │
├─────────────────────────────────────────┤
│  5. Classifier                          │
│     Linear → LayerNorm → GELU →         │
│     Linear → Output [batch, 2]          │
├─────────────────────────────────────────┤
│  Output: UP / DOWN 機率                  │
└─────────────────────────────────────────┘
```

**模型參數**
| 參數 | 值 |
|------|-----|
| 輸入維度 | 20 |
| 模型維度 | 64 |
| 注意力頭數 | 4 |
| Encoder 層數 | 2 |
| Dropout | 0.4 |
| 總參數量 | ~50K |

#### 4.1.2 LSTM 模型

```
┌─────────────────────────────────────────┐
│              CryptoLSTM                  │
├─────────────────────────────────────────┤
│  Input: [batch, 30, 20]                 │
│         (序列長度=30, 特徵數=20)          │
├─────────────────────────────────────────┤
│  1. Input Projection                    │
│     Linear(20 → 128) + LayerNorm + GELU │
├─────────────────────────────────────────┤
│  2. Bidirectional LSTM (x2 layers)      │
│     - 雙向 LSTM 捕捉前後文信息            │
│     - hidden_dim = 128                  │
│     - Dropout (0.3)                     │
├─────────────────────────────────────────┤
│  3. Layer Normalization                 │
│     正規化 LSTM 輸出                     │
├─────────────────────────────────────────┤
│  4. Attention Mechanism                 │
│     - 計算各時間步的注意力權重             │
│     - 加權聚合重要時間步                  │
├─────────────────────────────────────────┤
│  5. Classifier                          │
│     Linear(256→128) → LayerNorm → GELU →│
│     Linear(128→64) → LayerNorm → GELU → │
│     Linear(64→2) → Output               │
├─────────────────────────────────────────┤
│  Output: UP / DOWN 機率                  │
└─────────────────────────────────────────┘
```

**LSTM 模型參數**
| 參數 | 值 |
|------|-----|
| 輸入維度 | 20 |
| 隱藏層維度 | 128 |
| LSTM 層數 | 2 |
| 雙向 | 是 (Bidirectional) |
| Dropout | 0.3 |
| 總參數量 | ~400K |

**LSTM vs Transformer 設計差異**
| 特性 | Transformer | LSTM |
|------|-------------|------|
| 序列建模 | Self-Attention | 遞迴結構 |
| 並行性 | 高 | 低 |
| 長距離依賴 | 直接建模 | 通過隱狀態傳遞 |
| 參數量 | ~50K | ~400K |
| 位置編碼 | Sin/Cos | 隱式 |

### 4.2 訓練配置

**Transformer 配置**
```python
transformer_config = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.0002,
    "weight_decay": 0.1,      # L2 正則化
    "label_smoothing": 0.1,   # 標籤平滑
    "warmup_epochs": 15,      # 學習率預熱
    "patience": 30            # Early Stopping
}
```

**LSTM 配置**
```python
lstm_config = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.05,     # L2 正則化
    "label_smoothing": 0.1,   # 標籤平滑
    "warmup_epochs": 10,      # 學習率預熱
    "patience": 30            # Early Stopping
}
```

**共同訓練技巧**
- AdamW 優化器 + 權重衰減
- Cosine Annealing 學習率排程
- 梯度裁剪 (max_norm=1.0)
- 類別權重平衡

### 4.3 Flask API 部署

![Flask 伺服器啟動畫面](image/flask啟動畫面.png)

```python
# api/app.py - 主要端點

@app.route('/api/chat', methods=['POST'])
def chat():
    """智慧問答 - 支援自然語言查詢"""
    # 1. 解析用戶訊息
    # 2. 判斷意圖 (價格/預測/圖表)
    # 3. 調用對應工具
    # 4. 返回結果

@app.route('/api/predict/<symbol>')
def predict(symbol):
    """價格預測 API"""
    # 使用 Transformer 模型預測

@app.route('/api/price/<symbol>')
def get_price(symbol):
    """即時價格 API"""
    # 調用 CoinGecko API
```

**API 功能**
| 端點 | 功能 | 範例 |
|------|------|------|
| POST /api/chat | 自然語言問答 | "BTC 會漲嗎？" |
| GET /api/predict/BTC | 價格預測 | 返回 UP/DOWN |
| GET /api/price/BTC | 即時價格 | 返回 $97,250 |
| GET /api/history/BTC | 歷史走勢 | 返回圖表資料 |

![Web UI 介面](image/webui.png)

![問答 Demo - 預測功能](image/預測功能demo.png)

![問答 Demo - 價格查詢 1](image/價格查詢demo1.png)

![問答 Demo - 價格查詢 2](image/價格查詢demo2.png)

---

## 5. Optional：進階功能

### 5.1 LLMOps - LLM Agent 整合

**ReAct 模式 (Reasoning + Acting)**
```
User: "BTC 現在多少錢？明天會漲嗎？"
     │
     ▼
┌─────────────────────────────────────┐
│  LLM Agent (Ollama - llama3.2)     │
│  1. 理解用戶意圖                    │
│  2. 選擇工具: get_current_price    │
│  3. 選擇工具: predict_price        │
│  4. 解讀結果                        │
│  5. 用繁體中文回答                  │
└─────────────────────────────────────┘
     │
     ▼
"BTC 目前價格為 $97,250，24小時漲幅 2.15%。
 根據 Transformer 模型預測，明天有 64.53%
 的機率上漲，信心程度為「中」。"
```

**MCP Tools 定義** (`mcp/tools.py`)
```python
TOOLS = [
    "get_current_price",      # 即時價格
    "get_price_history",      # 歷史走勢
    "predict_price",          # ML 預測
    "get_technical_analysis", # 技術分析
    "analyze_sentiment"       # 情感分析
]
```

### 5.2 Docker 支援

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "-m", "api.app"]
```

---

## 6. 模型評估與成果展示

### 6.1 訓練結果

#### Transformer 模型結果

| 指標 | 數值 | 說明 |
|------|------|------|
| 最佳驗證準確率 | **57.99%** | Epoch 47 達到最佳 |
| 測試集準確率 | **54.63%** | 優於隨機猜測 (50%) |
| 測試集 Precision | 59.36% | 預測「漲」的準確度 |
| 測試集 F1 Score | 43.87% | 綜合評估指標 |
| 訓練時間 | 100.69 秒 | Early Stopping 於 77 epoch |

#### LSTM 模型結果

| 指標 | 數值 | 說明 |
|------|------|------|
| 最佳驗證準確率 | **57.35%** | Epoch 1 達到最佳 |
| 測試集準確率 | **51.60%** | 優於隨機猜測 (50%) |
| 測試集 Precision | 52.27% | 預測「漲」的準確度 |
| 測試集 Recall | 57.68% | 預測「漲」的召回率 |
| 測試集 F1 Score | 54.84% | 綜合評估指標 |
| 訓練時間 | 31.77 秒 | Early Stopping |

#### 模型比較

| 指標 | Transformer | LSTM | 勝出 |
|------|-------------|------|------|
| 最佳驗證準確率 | 57.99% | 57.35% | Transformer |
| 測試集準確率 | 54.63% | 51.60% | Transformer |
| 測試集 F1 Score | 43.87% | 54.84% | **LSTM** |
| 訓練時間 | 100.69 秒 | 31.77 秒 | **LSTM** |
| 參數量 | ~50K | ~400K | Transformer |

**結果分析**
- 金融市場預測本身極具挑戰性，學術研究普遍認為 55-60% 準確率已具參考價值
- **Transformer** 在測試集準確率上表現較佳，適合作為主要預測模型
- **LSTM** 在 F1 Score 上表現較佳，顯示其對於「漲」的預測更為平衡
- LSTM 訓練速度更快（約 3 倍），但參數量較大（約 8 倍）
- 兩種模型皆採用 Early Stopping 避免過擬合，並使用 Label Smoothing 增強泛化能力

### 6.2 預測示範

*（預測結果範例）*

```json
{
  "symbol": "BTC",
  "prediction": "UP",
  "confidence": 0.6453,
  "probabilities": {
    "UP": 0.6453,
    "DOWN": 0.3547
  },
  "model": "Transformer",
  "confidence_level": "中"
}
```

### 6.3 學習心得

**學到什麼？**

1. **完整 MLOps 流程**
   - 從資料收集、清理、特徵工程到模型訓練部署
   - 理解 ETL Pipeline 的重要性

2. **深度學習模型**
   - 學會將 NLP 常用的 Transformer 應用於時間序列預測
   - 理解 Attention 機制如何捕捉長期依賴
   - 實作雙向 LSTM 並結合 Attention 機制
   - 比較不同架構在金融預測上的表現差異

3. **API 設計與部署**
   - Flask RESTful API 設計
   - 前後端整合與 CORS 處理

4. **LLM Agent 整合**
   - Function Calling / Tool Use 機制
   - ReAct 模式實作

### 6.4 遇到的困難與解決方案

| 困難 | 解決方案 |
|------|----------|
| CoinCap API 停止服務 | 遷移至 CoinGecko API |
| SSL 憑證問題 (macOS) | 自訂 SSL Context |
| 模型過擬合 | 增加 Dropout、Label Smoothing |
| 特徵維度不匹配 | 統一訓練與預測的特徵處理 |
| LLM 不支援 Function Calling | 實作 Fallback 規則匹配 |

### 6.5 未來改進方向

1. **模型改進**
   - 加入更多特徵（新聞情感、鏈上數據）
   - 嘗試其他架構（Temporal Fusion Transformer、GRU）
   - 實作模型集成（Ensemble）結合 Transformer 與 LSTM 優勢

2. **系統擴展**
   - 支援更多幣種
   - 加入自動化訓練排程
   - 整合 MLflow 追蹤實驗

3. **部署優化**
   - 使用 Docker Compose 完整部署
   - 加入監控與告警機制

---

## 技術棧總覽

| 類別 | 技術 |
|------|------|
| 程式語言 | Python 3.11 |
| 深度學習 | PyTorch |
| Web 框架 | Flask |
| 前端 | HTML/CSS/JavaScript, Chart.js |
| LLM | Ollama (llama3.2) |
| 資料處理 | Pandas, NumPy |
| API | CoinGecko |
| 版本控制 | Git |
| 容器化 | Docker |

---


