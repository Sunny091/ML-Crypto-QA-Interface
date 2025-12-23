# Cryptocurrency Price Prediction System
## Automated Pipeline from Data Collection to Model Training and Deployment

**Machine Learning Project Report**

---

## 1. Project Title & Motivation

### Title
**Intelligent Cryptocurrency Q&A and Price Prediction System**
- An end-to-end MLOps project combining NLP and Deep Learning

### Motivation
1. **Market Demand**: Cryptocurrency markets are highly volatile; investors need real-time intelligent analysis tools
2. **Technical Integration**: Implement a complete MLOps workflow from data collection, cleaning, training to deployment
3. **Learning Objectives**:
   - Build ETL Pipeline for automated data processing
   - Train Transformer and LSTM deep learning models
   - Deploy RESTful API using Flask
   - Integrate LLM (Ollama) for natural language Q&A

---

## 2. System Architecture (ETL / Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      System Architecture Overview                    │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    User      │───▶│   Flask API  │───▶│  LLM Agent   │
│  (Web UI)    │◀───│   (app.py)   │◀───│  (Ollama)    │
└──────────────┘    └──────────────┘    └──────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐    ┌──────────────┐
                    │  MCP Tools   │    │ Tool Calling │
                    │  (Tool Layer)│◀───│  (Function)  │
                    └──────────────┘    └──────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ Real-time  │   │ Historical │   │    ML      │
   │   Price    │   │   Price    │   │ Prediction │
   │ CoinGecko  │   │   Charts   │   │Transformer │
   └────────────┘   └────────────┘   │   & LSTM   │
                                      └────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       ETL Pipeline                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐       │
│  │ Extract │────▶│Transform│────▶│  Load   │────▶│  Train  │       │
│  │ Scraper │     │ Feature │     │ Dataset │     │  Model  │       │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘       │
│       │               │               │               │             │
│       ▼               ▼               ▼               ▼             │
│  CoinGecko API   20 Technical    Train/Val/Test   Transformer       │
│  Kaggle Dataset  Indicators      70/15/15 Split   LSTM Model        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Project Directory Structure
```
ML-Crypto-QA-Interface/
├── api/                    # Flask API Service
│   ├── app.py             # Main Application
│   └── static/            # Frontend UI
├── data/                   # Data Layer
│   ├── raw/               # Raw Data (Kaggle)
│   └── scrapers/          # Data Scrapers (CoinGecko)
├── etl/                    # ETL Pipeline
│   ├── extract.py         # Data Extraction
│   └── transform.py       # Feature Engineering
├── models/                 # Model Layer
│   ├── transformer/       # Transformer Model
│   │   ├── model.py      # Model Architecture
│   │   ├── train.py      # Training Script
│   │   └── predictor.py  # Predictor
│   ├── lstm/              # LSTM Model
│   │   ├── model.py      # Model Architecture
│   │   ├── train.py      # Training Script
│   │   └── predictor.py  # Predictor
│   └── artifacts/         # Model Files
│       ├── transformer.pt # Transformer Weights
│       └── lstm.pt        # LSTM Weights
├── orchestrator/           # LLM Agent
│   └── agent.py           # ReAct Pattern Agent
├── mcp/                    # MCP Tool Definitions
│   └── tools.py           # Tool Implementations
└── evaluation/             # Evaluation Results
    └── results/           # Training Reports
```

---

## 3. Data Layer: DataOps

### 3.1 Data Scraping (Extract)

**Data Sources**
| Source | Purpose | Data Volume |
|--------|---------|-------------|
| CoinGecko API | Real-time prices, historical trends | Real-time |
| Kaggle Dataset | Model training | 4,371 records (BTC) |

**CoinGecko API Integration** (`data/scrapers/coincap_client.py`)
```python
# Supported Cryptocurrencies
SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana"
}

# API Endpoints
- /simple/price             # Real-time price
- /coins/{id}/market_chart  # Historical trends
```

### 3.2 Data Cleaning & Feature Engineering (Transform)

**20 Technical Indicator Features**

| Category | Feature Names | Description |
|----------|---------------|-------------|
| Returns | return_1d, 3d, 5d, 10d | Multi-scale returns |
| Returns | log_return_1d | Log returns |
| RSI | rsi_7, rsi_14, rsi_21 | Relative Strength Index |
| Volatility | volatility_5d, 10d, 20d | Price volatility |
| MA | ma_ratio_5, ma_ratio_20 | Moving average ratio |
| MA | ma_cross_5_20 | MA crossover signal |
| MACD | macd_hist | MACD histogram |
| BB | bb_width, bb_position | Bollinger Bands |
| Other | high_low_ratio | High-low ratio |
| Other | close_position | Close price position |
| Other | volume_ratio | Volume ratio |

**Data Processing Pipeline**
```python
# 1. Load raw data
df = load_kaggle_data("BTC")  # 4,371 records

# 2. Calculate technical indicators
df = add_features(df)  # 20 features

# 3. Normalization
features = normalize(features)  # Z-score

# 4. Split dataset
Train: 70% (3,029 records)
Val:   15% (626 records)
Test:  15% (626 records)
```

---

## 4. Model Layer: MLOps

### 4.1 Model Architecture

This project implements two deep learning models for cryptocurrency price prediction:

#### 4.1.1 Transformer Model

```
┌─────────────────────────────────────────┐
│           CryptoTransformer             │
├─────────────────────────────────────────┤
│  Input: [batch, 30, 20]                 │
│         (seq_len=30, features=20)       │
├─────────────────────────────────────────┤
│  1. Input Projection                    │
│     Linear(20 → 64) + LayerNorm + GELU  │
├─────────────────────────────────────────┤
│  2. Positional Encoding                 │
│     Sinusoidal position encoding        │
├─────────────────────────────────────────┤
│  3. Transformer Encoder (x2 layers)     │
│     - Multi-Head Attention (4 heads)    │
│     - Feed Forward (128 dim)            │
│     - Dropout (0.4)                     │
├─────────────────────────────────────────┤
│  4. Attention Pooling                   │
│     Weighted average of all timesteps   │
├─────────────────────────────────────────┤
│  5. Classifier                          │
│     Linear → LayerNorm → GELU →         │
│     Linear → Output [batch, 2]          │
├─────────────────────────────────────────┤
│  Output: UP / DOWN probabilities        │
└─────────────────────────────────────────┘
```

**Model Parameters**
| Parameter | Value |
|-----------|-------|
| Input Dimension | 20 |
| Model Dimension | 64 |
| Attention Heads | 4 |
| Encoder Layers | 2 |
| Dropout | 0.4 |
| Total Parameters | ~50K |

#### 4.1.2 LSTM Model

```
┌─────────────────────────────────────────┐
│              CryptoLSTM                  │
├─────────────────────────────────────────┤
│  Input: [batch, 30, 20]                 │
│         (seq_len=30, features=20)       │
├─────────────────────────────────────────┤
│  1. Input Projection                    │
│     Linear(20 → 128) + LayerNorm + GELU │
├─────────────────────────────────────────┤
│  2. Bidirectional LSTM (x2 layers)      │
│     - Bidirectional for context         │
│     - hidden_dim = 128                  │
│     - Dropout (0.3)                     │
├─────────────────────────────────────────┤
│  3. Layer Normalization                 │
│     Normalize LSTM output               │
├─────────────────────────────────────────┤
│  4. Attention Mechanism                 │
│     - Compute attention weights         │
│     - Aggregate important timesteps     │
├─────────────────────────────────────────┤
│  5. Classifier                          │
│     Linear(256→128) → LayerNorm → GELU →│
│     Linear(128→64) → LayerNorm → GELU → │
│     Linear(64→2) → Output               │
├─────────────────────────────────────────┤
│  Output: UP / DOWN probabilities        │
└─────────────────────────────────────────┘
```

**LSTM Model Parameters**
| Parameter | Value |
|-----------|-------|
| Input Dimension | 20 |
| Hidden Dimension | 128 |
| LSTM Layers | 2 |
| Bidirectional | Yes |
| Dropout | 0.3 |
| Total Parameters | ~400K |

**LSTM vs Transformer Design Differences**
| Feature | Transformer | LSTM |
|---------|-------------|------|
| Sequence Modeling | Self-Attention | Recurrent |
| Parallelism | High | Low |
| Long-range Dependencies | Direct modeling | Via hidden state |
| Parameters | ~50K | ~400K |
| Position Encoding | Sin/Cos | Implicit |

### 4.2 Training Configuration

**Transformer Configuration**
```python
transformer_config = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.0002,
    "weight_decay": 0.1,      # L2 regularization
    "label_smoothing": 0.1,   # Label smoothing
    "warmup_epochs": 15,      # Learning rate warmup
    "patience": 30            # Early Stopping
}
```

**LSTM Configuration**
```python
lstm_config = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.05,     # L2 regularization
    "label_smoothing": 0.1,   # Label smoothing
    "warmup_epochs": 10,      # Learning rate warmup
    "patience": 30            # Early Stopping
}
```

**Common Training Techniques**
- AdamW optimizer + weight decay
- Cosine Annealing learning rate schedule
- Gradient clipping (max_norm=1.0)
- Class weight balancing

### 4.3 Flask API Deployment

![Flask Server Startup](image/flask啟動畫面.png)

```python
# api/app.py - Main Endpoints

@app.route('/api/chat', methods=['POST'])
def chat():
    """Intelligent Q&A - Natural language queries"""
    # 1. Parse user message
    # 2. Determine intent (price/predict/chart)
    # 3. Call corresponding tool
    # 4. Return result

@app.route('/api/predict/<symbol>')
def predict(symbol):
    """Price Prediction API"""
    # Use Transformer model for prediction

@app.route('/api/price/<symbol>')
def get_price(symbol):
    """Real-time Price API"""
    # Call CoinGecko API
```

**API Features**
| Endpoint | Function | Example |
|----------|----------|---------|
| POST /api/chat | Natural language Q&A | "Will BTC go up?" |
| GET /api/predict/BTC | Price prediction | Returns UP/DOWN |
| GET /api/price/BTC | Real-time price | Returns $97,250 |
| GET /api/history/BTC | Historical trends | Returns chart data |

![Web UI Interface](image/webui.png)

![Q&A Demo - Prediction Feature](image/預測功能demo.png)

![Q&A Demo - Price Query 1](image/價格查詢demo1.png)

![Q&A Demo - Price Query 2](image/價格查詢demo2.png)

---

## 5. Optional: Advanced Features

### 5.1 LLMOps - LLM Agent Integration

**ReAct Pattern (Reasoning + Acting)**
```
User: "What's the current BTC price? Will it go up tomorrow?"
     │
     ▼
┌─────────────────────────────────────┐
│  LLM Agent (Ollama - llama3.2)     │
│  1. Understand user intent          │
│  2. Select tool: get_current_price  │
│  3. Select tool: predict_price      │
│  4. Interpret results               │
│  5. Respond in natural language     │
└─────────────────────────────────────┘
     │
     ▼
"BTC current price is $97,250, up 2.15% in 24h.
 Based on Transformer model prediction, there's
 a 64.53% probability of rising tomorrow,
 with medium confidence level."
```

**MCP Tools Definition** (`mcp/tools.py`)
```python
TOOLS = [
    "get_current_price",      # Real-time price
    "get_price_history",      # Historical trends
    "predict_price",          # ML prediction
    "get_technical_analysis", # Technical analysis
    "analyze_sentiment"       # Sentiment analysis
]
```

### 5.2 Docker Support

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

## 6. Model Evaluation & Results

### 6.1 Training Results

#### Transformer Model Results

| Metric | Value | Description |
|--------|-------|-------------|
| Best Validation Accuracy | **57.99%** | Achieved at Epoch 47 |
| Test Accuracy | **54.63%** | Better than random guess (50%) |
| Test Precision | 59.36% | Accuracy for "UP" predictions |
| Test F1 Score | 43.87% | Combined evaluation metric |
| Training Time | 100.69 sec | Early Stopping at epoch 77 |

#### LSTM Model Results

| Metric | Value | Description |
|--------|-------|-------------|
| Best Validation Accuracy | **57.35%** | Achieved at Epoch 1 |
| Test Accuracy | **51.60%** | Better than random guess (50%) |
| Test Precision | 52.27% | Accuracy for "UP" predictions |
| Test Recall | 57.68% | Recall for "UP" predictions |
| Test F1 Score | 54.84% | Combined evaluation metric |
| Training Time | 31.77 sec | Early Stopping |

#### Model Comparison

| Metric | Transformer | LSTM | Winner |
|--------|-------------|------|--------|
| Best Validation Accuracy | 57.99% | 57.35% | Transformer |
| Test Accuracy | 54.63% | 51.60% | Transformer |
| Test F1 Score | 43.87% | 54.84% | **LSTM** |
| Training Time | 100.69 sec | 31.77 sec | **LSTM** |
| Parameters | ~50K | ~400K | Transformer |

**Results Analysis**
- Financial market prediction is inherently challenging; 55-60% accuracy is considered valuable in academic research
- **Transformer** achieves better test accuracy, suitable as the primary prediction model
- **LSTM** achieves better F1 Score, indicating more balanced predictions for "UP"
- LSTM trains faster (~3x), but has more parameters (~8x)
- Both models use Early Stopping to prevent overfitting and Label Smoothing for better generalization

### 6.2 Prediction Demo

*(Prediction Result Example)*

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
  "confidence_level": "Medium"
}
```

### 6.3 Lessons Learned

**What I Learned**

1. **Complete MLOps Workflow**
   - From data collection, cleaning, feature engineering to model training and deployment
   - Understanding the importance of ETL Pipeline

2. **Deep Learning Models**
   - Applied Transformer (commonly used in NLP) to time series prediction
   - Understanding how Attention mechanism captures long-term dependencies
   - Implemented Bidirectional LSTM with Attention mechanism
   - Compared different architectures for financial prediction

3. **API Design & Deployment**
   - Flask RESTful API design
   - Frontend-backend integration and CORS handling

4. **LLM Agent Integration**
   - Function Calling / Tool Use mechanism
   - ReAct pattern implementation

### 6.4 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| CoinCap API discontinued | Migrated to CoinGecko API |
| SSL certificate issues (macOS) | Custom SSL Context |
| Model overfitting | Increased Dropout, Label Smoothing |
| Feature dimension mismatch | Unified feature processing for training and prediction |
| LLM doesn't support Function Calling | Implemented fallback rule matching |

### 6.5 Future Improvements

1. **Model Enhancement**
   - Add more features (news sentiment, on-chain data)
   - Try other architectures (Temporal Fusion Transformer, GRU)
   - Implement model ensemble combining Transformer and LSTM strengths

2. **System Expansion**
   - Support more cryptocurrencies
   - Add automated training scheduling
   - Integrate MLflow for experiment tracking

3. **Deployment Optimization**
   - Use Docker Compose for complete deployment
   - Add monitoring and alerting mechanisms

---

## Technology Stack Overview

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.11 |
| Deep Learning | PyTorch |
| Web Framework | Flask |
| Frontend | HTML/CSS/JavaScript, Chart.js |
| LLM | Ollama (llama3.2) |
| Data Processing | Pandas, NumPy |
| API | CoinGecko |
| Version Control | Git |
| Containerization | Docker |

---

## Q&A

Thank you for listening!

