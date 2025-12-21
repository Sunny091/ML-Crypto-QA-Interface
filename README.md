# MCP Orchestrator System

**Natural Language Control for Cryptocurrency Analysis**

> A modular system for cryptocurrency analysis using natural language. Built with an MCP (Model Context Protocol) architecture separating control plane (Orchestrator) from execution plane (Tool Server).

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Control Plane (Chat UI + Ollama Orchestrator) | ✅ Complete |
| Phase 2 | Execution Plane (fastmcp Tool Server) | ✅ Complete |
| Phase 3 | Integration (Orchestrator calls Tools) | ✅ Complete |
| Phase 4 | News Analysis (price+text model) | ✅ Complete |

---

# Phase 1: Control Plane

> Frontend Chat UI + Backend calling Ollama LLM as an orchestrator that outputs strict JSON plans.

## Architecture

```
User (Chat UI)
    ↓
Backend API (/api/chat)
    ↓
LLM Orchestrator (via Ollama)
    ↓
JSON Plan (no execution)
```

## Tech Stack

### Frontend
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Chat-style UI

### Backend
- Node.js + Express
- TypeScript
- Zod for schema validation
- Fetch API for Ollama calls

### LLM
- Local Ollama
- Model configurable via `.env`

## Quick Start

### Prerequisites
- Node.js 18+
- Ollama installed and running
- Model downloaded (e.g., `ollama pull gpt-oss:20b`)

### Setup

1. **Clone and configure environment**
   ```bash
   cp .env.example .env
   # Edit .env to match your setup
   ```

2. **Install dependencies**
   ```bash
   # Backend
   cd server
   npm install

   # Frontend
   cd ../frontend
   npm install
   ```

3. **Start the servers**
   ```bash
   # Terminal 1: Start backend
   cd server
   npm run dev

   # Terminal 2: Start frontend
   cd frontend
   npm run dev
   ```

4. **Access the UI**
   - Frontend: http://localhost:3000
   - Backend: http://localhost:3001

## Environment Variables

All configuration is done via `.env` file in the project root.

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama API URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name | `gpt-oss:20b` |
| `OLLAMA_TIMEOUT_MS` | Request timeout | `60000` |
| `SERVER_HOST` | Backend host | `localhost` |
| `SERVER_PORT` | Backend port | `3001` |
| `NEXT_PUBLIC_API_BASE_URL` | Backend URL for frontend | `http://localhost:3001` |
| `APP_ENV` | Environment mode | `development` |
| `LOG_LEVEL` | Logging level | `debug` |

## API Endpoints

### POST /api/chat

Send a message to the orchestrator.

**Request:**
```json
{
  "message": "幫我預測 BTC 現在的隔日漲跌"
}
```

**Response:**
```json
{
  "plan": {
    "task": "predict",
    "symbol": "BTC",
    "as_of": "now",
    "horizon": "1d",
    "use_text_features": false,
    "news_text_provided": false,
    "news_text": null
  },
  "assistant_text": "已收到預測請求，目標幣種：BTC。計畫已生成，等待執行。"
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "gpt-oss:20b",
  "env": "development"
}
```

## JSON Plan Schema

The LLM orchestrator outputs JSON following this strict schema:

```json
{
  "task": "predict" | "analyze_news" | "help" | "unknown",
  "symbol": "BTC" | "ETH" | "SOL" | null,
  "as_of": "now" | "YYYY-MM-DD" | null,
  "horizon": "1d" | "7d" | null,
  "use_text_features": true | false,
  "news_text_provided": true | false,
  "news_text": "string | null"
}
```

### Schema Rules
- No symbol mentioned → `symbol: null`
- No date mentioned → `as_of: "now"`
- News text provided → `news_text_provided: true`
- `news_text` max 2000 characters (truncated if longer)
- Cannot understand → `task: "unknown"`

## Test Cases

### Test 1: Basic Prediction Request
**Input:**
```
幫我預測 BTC 現在的隔日漲跌
```

**Expected Plan:**
```json
{
  "task": "predict",
  "symbol": "BTC",
  "as_of": "now",
  "horizon": "1d",
  "use_text_features": false,
  "news_text_provided": false,
  "news_text": null
}
```

### Test 2: Prediction with Specific Date
**Input:**
```
預測 2023-06-01 的 ETH 明天走勢
```

**Expected Plan:**
```json
{
  "task": "predict",
  "symbol": "ETH",
  "as_of": "2023-06-01",
  "horizon": "1d",
  "use_text_features": false,
  "news_text_provided": false,
  "news_text": null
}
```

### Test 3: News Analysis with Text
**Input:**
```
分析這則新聞對 BTC 的影響：

Bitcoin Price Surges Past $50,000 as Institutional Investors Show Renewed Interest

Major financial institutions have announced significant Bitcoin purchases, signaling a shift in institutional sentiment. The surge comes amid growing concerns about inflation and the weakening of traditional currencies.
```

**Expected Plan:**
```json
{
  "task": "analyze_news",
  "symbol": "BTC",
  "as_of": "now",
  "horizon": null,
  "use_text_features": true,
  "news_text_provided": true,
  "news_text": "Bitcoin Price Surges Past $50,000 as Institutional Investors Show Renewed Interest..."
}
```

### Test 4: Help Request
**Input:**
```
這個系統能做什麼？
```

**Expected Plan:**
```json
{
  "task": "help",
  "symbol": null,
  "as_of": null,
  "horizon": null,
  "use_text_features": false,
  "news_text_provided": false,
  "news_text": null
}
```

### Test 5: Unknown Intent
**Input:**
```
今天天氣如何？
```

**Expected Plan:**
```json
{
  "task": "unknown",
  "symbol": null,
  "as_of": null,
  "horizon": null,
  "use_text_features": false,
  "news_text_provided": false,
  "news_text": null
}
```

## Project Structure

```
.
├── .env.example          # Environment template
├── .env                  # Local environment (git ignored)
├── README.md             # This file
├── server/               # Phase 1: Node.js Backend (Orchestrator)
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── index.ts          # Server entry point
│       ├── config/
│       │   └── env.ts        # Single source of env vars
│       ├── routes/
│       │   └── chat.ts       # Chat API route
│       ├── services/
│       │   └── ollama.ts     # Ollama integration
│       └── schemas/
│           └── plan.ts       # JSON schema validation
├── frontend/             # Phase 1: Next.js Frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env.local
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx      # Chat UI
│       │   └── globals.css
│       ├── components/
│       │   ├── ChatMessage.tsx
│       │   ├── ChatInput.tsx
│       │   └── JsonDisplay.tsx
│       └── lib/
│           ├── config.ts     # Frontend config
│           └── api.ts        # API client
└── mcp_server/           # Phase 2: Python MCP Tool Server
    ├── requirements.txt
    ├── server.py             # fastmcp server
    ├── train_model.py        # Model training
    ├── test_tool.py          # Test script
    ├── config/               # Environment config
    ├── data/                 # Mock data generation
    ├── features/             # Feature engineering
    ├── models/               # Model loading
    └── tools/                # Tool schemas
```

## Key Design Decisions

### LLM as Orchestrator Only
- The LLM does NOT generate predictions
- The LLM does NOT provide natural language answers
- The LLM ONLY extracts structured intent from user input

### Single Source of Truth for Config
- All `process.env` reads happen in `server/src/config/env.ts`
- No hardcoded URLs, ports, or model names anywhere
- Startup fails fast if required env vars are missing

### JSON Validation with Retry
- Uses Zod for schema validation
- Automatically retries up to 2 times on parse failures
- Returns raw output for debugging in development mode

## What Phase 1 Does NOT Include

- ❌ Price predictions
- ❌ Model training
- ❌ fastmcp integration
- ❌ Real-time price data
- ❌ Historical data analysis

These will be added in Phase 2 and Phase 3.

## Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if model is available
ollama list
```

### Backend Not Starting
```bash
# Check environment variables
cat .env

# Verify required vars are set
```

### Frontend API Error
```bash
# Verify backend is running
curl http://localhost:3001/api/health

# Check CORS settings
```

---

**Phase 1 Rule:** This phase is about control plane, not intelligence.

---

# Phase 2: Execution Plane

> fastmcp Tool Server providing deterministic price prediction using LightGBM.

## Phase 2 Architecture

```
Tool Server (fastmcp)
    ↓
predict_price_price_only tool
    ↓
Feature Engineering → LightGBM Model → Prediction
```

**Important:** Phase 2 is standalone. It does NOT connect to the Orchestrator (that's Phase 3).

## Phase 2 Tech Stack

- Python 3.10+
- fastmcp (MCP protocol implementation)
- LightGBM (ML model)
- Pydantic (schema validation)
- pandas, numpy (data processing)

## Phase 2 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Setup

1. **Install Python dependencies**
   ```bash
   cd mcp_server
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python train_model.py
   ```

3. **Test the pipeline**
   ```bash
   python test_tool.py
   ```

4. **Start the MCP server**
   ```bash
   python server.py
   ```

## Phase 2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_SERVER_NAME` | Server name for MCP | `crypto-predictor` |
| `MCP_SERVER_HOST` | Server host | `localhost` |
| `MCP_SERVER_PORT` | Server port | `4000` |
| `MODEL_DIR` | Directory for model files | `./models` |
| `PRICE_ONLY_MODEL_NAME` | Model filename | `price_only_lgbm.pkl` |
| `PRICE_DATA_SOURCE` | Data source (mock/api/file) | `mock` |

## MCP Tool: predict_price_price_only

Predict next-day price movement using price data only.

### Input Schema

```json
{
  "symbol": "BTC" | "ETH" | "SOL",
  "as_of": "YYYY-MM-DD" | "now"
}
```

### Output Schema

```json
{
  "symbol": "BTC",
  "as_of": "2024-06-15",
  "prediction": "UP" | "DOWN",
  "prob_up": 0.5138,
  "model_variant": "price_only",
  "features_used": ["return_1d", "rsi_14", "volatility_7d"]
}
```

### Features Used

| Feature | Description |
|---------|-------------|
| `return_1d` | 1-day price return (close-to-close) |
| `rsi_14` | 14-day Relative Strength Index |
| `volatility_7d` | 7-day rolling volatility (std of returns) |

## Phase 2 Test Examples

### Test via Python

```python
from tools import PredictPriceInput
from data import get_price_data
from features import get_latest_features
from models import get_predictor

# Initialize
predictor = get_predictor()

# Get data and features
df = get_price_data("BTC", as_of="2024-06-15", lookback_days=60)
features = get_latest_features(df)

# Predict
prediction, prob_up = predictor.predict(features)
print(f"Prediction: {prediction}, Probability UP: {prob_up:.4f}")
```

### Test via MCP (when server is running)

The server uses STDIO transport by default. For testing, use the `test_tool.py` script which tests the pipeline directly.

## Phase 2 Project Structure

```
mcp_server/
├── requirements.txt      # Python dependencies
├── server.py             # Main fastmcp server
├── train_model.py        # Model training script
├── test_tool.py          # Test script
├── config/
│   └── __init__.py       # Environment configuration
├── data/
│   └── __init__.py       # Mock data generation
├── features/
│   └── __init__.py       # Feature engineering
├── models/
│   └── __init__.py       # Model loading/prediction
│   └── price_only_lgbm.pkl  # Trained model (after training)
└── tools/
    └── __init__.py       # Tool schemas
```

## Phase 2 Design Decisions

### Deterministic Pipeline
- All feature calculations are reproducible
- Mock data uses fixed random seed
- No external API dependencies (Phase 2)

### Single Responsibility
- fastmcp server ONLY executes tools
- No natural language understanding
- No decision making about when to use tools

### Model Simplicity
- LightGBM binary classifier
- 3 simple technical features
- Accuracy is not the goal (pipeline correctness is)

## What Phase 2 Does NOT Include

- ❌ Connection to Orchestrator
- ❌ News/text analysis
- ❌ Multiple model variants
- ❌ Real-time price data
- ❌ Natural language processing

These will be added in Phase 3 and beyond.

## Troubleshooting Phase 2

### Model Not Found
```bash
# Train the model first
cd mcp_server
python train_model.py
```

### Import Errors
```bash
# Make sure you're in the mcp_server directory
cd mcp_server
python -c "from config import env; print(env)"
```

### Test Pipeline
```bash
# Run the test script to verify everything works
python test_tool.py
```

---

**Phase 2 Rule:** This phase is about deterministic execution, not intelligence.

---

# Phase 3: Integration Layer

> Glue layer that routes Orchestrator JSON plans to fastmcp tool execution.

## Phase 3 Architecture

```
User (Chat UI)
    ↓
Backend API (/api/chat)
    ↓
LLM Orchestrator (JSON plan)
    ↓
Planner/Router (executePlan.ts)
    ↓
MCP Client (mcpClient.ts)
    ↓
HTTP Server (http_server.py)
    ↓
Tool Execution (predict_price_price_only)
    ↓
Tool Result (JSON)
    ↓
User-facing Response
```

## Phase 3 Quick Start

### Prerequisites
- All Phase 1 and Phase 2 setup complete
- Model trained (`python train_model.py`)

### Start All Services

```bash
# Terminal 1: MCP HTTP Server (Tool Server)
cd mcp_server
python http_server.py

# Terminal 2: Node.js Backend (Orchestrator + Router)
cd server
npm run dev

# Terminal 3: Next.js Frontend
cd frontend
npm run dev
```

### Access
- Frontend: http://localhost:3000
- Backend: http://localhost:3001
- MCP Server: http://localhost:4000

## Phase 3 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_CLIENT_HOST` | MCP Tool Server URL | `http://localhost:4000` |
| `MCP_CLIENT_TIMEOUT_MS` | Request timeout | `30000` |

## Phase 3 API Response Format

### POST /api/chat (Updated)

**Request:**
```json
{
  "message": "幫我預測 BTC 現在的隔日漲跌"
}
```

**Response (Tool Called):**
```json
{
  "plan": {
    "task": "predict",
    "symbol": "BTC",
    "as_of": "now",
    "horizon": "1d",
    "use_text_features": false,
    "news_text_provided": false,
    "news_text": null
  },
  "tool_called": true,
  "tool_name": "predict_price_price_only",
  "tool_result": {
    "symbol": "BTC",
    "as_of": "2024-12-18",
    "prediction": "UP",
    "prob_up": 0.5342,
    "model_variant": "price_only",
    "features_used": ["return_1d", "rsi_14", "volatility_7d"]
  },
  "assistant_text": "使用價格模型預測 BTC：預測隔日上漲（信心度 53.4%）。模型僅使用價格特徵，未使用新聞資料。"
}
```

**Response (Tool Not Called):**
```json
{
  "plan": {
    "task": "predict",
    "symbol": "BTC",
    "use_text_features": true,
    ...
  },
  "tool_called": false,
  "reason": "Text features (news analysis) are not supported in Phase 3. Only price-based prediction is available.",
  "assistant_text": "Text features (news analysis) are not supported in Phase 3..."
}
```

## Phase 3 Test Cases

### Case 1: Successful Prediction
**Input:**
```
幫我預測 BTC 現在的隔日漲跌
```

**Expected:**
- `plan.task` = "predict"
- `tool_called` = true
- `tool_result` contains prediction

### Case 2: Prediction Rejected (News Analysis)
**Input:**
```
根據這篇新聞，預測 BTC：
[貼上新聞內容]
```

**Expected:**
- `plan.use_text_features` = true
- `tool_called` = false
- `reason` explains Phase 3 limitation

### Case 3: Help Request
**Input:**
```
這個系統能做什麼？
```

**Expected:**
- `plan.task` = "help"
- `tool_called` = false
- `assistant_text` explains capabilities

## Phase 3 Project Structure (New Files)

```
server/
├── src/
│   ├── planner/
│   │   └── executePlan.ts    # Plan router (Phase 3)
│   └── services/
│       └── mcpClient.ts      # MCP HTTP client (Phase 3)

mcp_server/
├── http_server.py            # HTTP wrapper for tools (Phase 3)
```

## Phase 3 Planner Logic

The planner (`executePlan.ts`) decides whether to call a tool:

| Condition | Action |
|-----------|--------|
| `task === "predict"` AND `symbol !== null` AND `use_text_features === false` | Call `predict_price_price_only` |
| `task === "predict"` AND `use_text_features === true` | Reject (Phase 4 feature) |
| `task === "help"` or `task === "unknown"` | No tool call, return message |
| `symbol === null` | Reject (missing parameter) |

## Phase 3 Design Decisions

### Separation of Concerns
- **Orchestrator (LLM)**: Only understands natural language, outputs JSON
- **Planner (Backend)**: Only routes plans, no NLU
- **Tool Server (Python)**: Only executes, no decisions

### HTTP Bridge
- Phase 2 fastmcp uses STDIO transport
- Phase 3 adds HTTP wrapper (`http_server.py`) for Node.js integration
- Same tool logic, different transport

### No Prediction Modification
- Backend does NOT modify tool outputs
- `prob_up` is exactly what the model returns
- Frontend displays raw tool results

## What Phase 3 Does NOT Include

- ❌ News/text analysis (use_text_features)
- ❌ Multiple tool routing
- ❌ Tool result caching
- ❌ Retry logic for tool calls

These may be added in Phase 4 and beyond.

## Troubleshooting Phase 3

### MCP Server Connection Error
```bash
# Check if MCP HTTP server is running
curl http://localhost:4000/health

# If not running, start it:
cd mcp_server
python http_server.py
```

### Tool Call Failed
```bash
# Test tool directly
curl -X POST http://localhost:4000/tools/predict_price_price_only \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC", "as_of": "2024-06-15"}'
```

### Backend Not Calling Tools
```bash
# Check backend logs (should show [Planner] and [MCPClient] messages)
# Make sure APP_ENV=development in .env for verbose logging
```

---

**Phase 3 Rule:** This phase is about routing, not reasoning.

---

# Phase 4: News Analysis

> Add news sentiment analysis and price+text prediction model.

## Phase 4 Architecture

```
User (Chat UI)
    ↓
Backend API (/api/chat)
    ↓
LLM Orchestrator (JSON plan with news_text)
    ↓
Planner/Router (executePlan.ts)
    ↓
[Route based on use_text_features]
    ├─ use_text_features=false → predict_price_price_only (Phase 2)
    └─ use_text_features=true  → predict_price_with_text (Phase 4)
    ↓
Tool Result (JSON with sentiment_score)
    ↓
User-facing Response
```

## Phase 4 Features

### New Tool: predict_price_with_text

Predict next-day price movement using price data AND news sentiment.

#### Input Schema

```json
{
  "symbol": "BTC" | "ETH" | "SOL",
  "as_of": "YYYY-MM-DD" | "now",
  "news_text": "string"
}
```

#### Output Schema

```json
{
  "symbol": "BTC",
  "as_of": "2024-06-15",
  "prediction": "UP" | "DOWN",
  "prob_up": 0.6334,
  "model_variant": "price_plus_text",
  "features_used": ["return_1d", "rsi_14", "volatility_7d", "sentiment_score", "news_present"],
  "sentiment_score": 0.8146
}
```

### Sentiment Analysis

Phase 4 includes a sentiment analysis module with two modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `finbert` | FinBERT transformer model (ProsusAI/finbert) | Production, accurate |
| `mock` | Rule-based keyword matching | Testing, fast |

Sentiment is returned as a score from -1.0 (bearish) to +1.0 (bullish).

### Price+Text Features

The price+text model uses 5 features:

| Feature | Description |
|---------|-------------|
| `return_1d` | 1-day price return (close-to-close) |
| `rsi_14` | 14-day Relative Strength Index |
| `volatility_7d` | 7-day rolling volatility |
| `sentiment_score` | News sentiment (-1.0 to +1.0) |
| `news_present` | Binary flag (0.0 or 1.0) |

## Phase 4 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEWS_MODEL_TYPE` | Sentiment model (`finbert` or `mock`) | `finbert` |
| `NEWS_MAX_LENGTH` | Max news text length | `2000` |
| `PRICE_TEXT_MODEL_NAME` | Model filename | `price_text_lgbm.pkl` |

## Phase 4 Quick Start

### Train Both Models

```bash
cd mcp_server

# Train price-only model (Phase 2)
python train_model.py

# Train price+text model (Phase 4)
python train_model.py --with-text

# Or train both at once
python train_model.py --all
```

### Test Price+Text Tool

```python
from tools import PredictPriceWithTextInput
from data import get_price_data
from features import get_price_text_features
from models import get_text_predictor

# Get data
df = get_price_data("BTC", as_of="2024-06-01", lookback_days=60)

# Get combined features (price + news sentiment)
news_text = "Bitcoin surges to new highs amid ETF approval"
features = get_price_text_features(df, news_text)
print(f"Features: {features}")

# Predict
predictor = get_text_predictor()
prediction, prob_up = predictor.predict(features)
print(f"Prediction: {prediction}, Probability UP: {prob_up:.4f}")
```

### Test via HTTP

```bash
curl -X POST http://localhost:4000/tools/predict_price_with_text \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "as_of": "2024-06-01",
    "news_text": "Bitcoin surges to new highs amid ETF approval"
  }'
```

**Response:**
```json
{
  "symbol": "BTC",
  "as_of": "2024-06-01",
  "prediction": "UP",
  "prob_up": 0.6334,
  "model_variant": "price_plus_text",
  "features_used": ["return_1d", "rsi_14", "volatility_7d", "sentiment_score", "news_present"],
  "sentiment_score": 0.8146
}
```

## Phase 4 API Response Format

### POST /api/chat (with News Text)

**Request:**
```json
{
  "message": "根據這則新聞預測 BTC：Bitcoin surges past $50,000 as institutions buy"
}
```

**Response:**
```json
{
  "plan": {
    "task": "predict",
    "symbol": "BTC",
    "as_of": "now",
    "horizon": "1d",
    "use_text_features": true,
    "news_text_provided": true,
    "news_text": "Bitcoin surges past $50,000 as institutions buy"
  },
  "tool_called": true,
  "tool_name": "predict_price_with_text",
  "tool_result": {
    "symbol": "BTC",
    "as_of": "2024-12-18",
    "prediction": "UP",
    "prob_up": 0.6334,
    "model_variant": "price_plus_text",
    "features_used": ["return_1d", "rsi_14", "volatility_7d", "sentiment_score", "news_present"],
    "sentiment_score": 0.8146
  },
  "assistant_text": "使用價格+新聞模型預測 BTC：預測隔日上漲（信心度 63.3%）。新聞情緒分析：正面（0.81）。"
}
```

## Phase 4 Planner Logic

| Condition | Action |
|-----------|--------|
| `task === "predict"` AND `use_text_features === false` | Call `predict_price_price_only` |
| `task === "predict"` AND `use_text_features === true` AND `news_text` provided | Call `predict_price_with_text` |
| `task === "predict"` AND `use_text_features === true` AND no `news_text` | Reject (missing news text) |

## Phase 4 Project Structure (New/Updated Files)

```
mcp_server/
├── news/
│   └── __init__.py           # Sentiment analysis (NEW)
├── features/
│   └── __init__.py           # Price+text features (UPDATED)
├── models/
│   ├── __init__.py           # PriceTextPredictor (UPDATED)
│   └── price_text_lgbm.pkl   # Trained model (NEW)
├── tools/
│   └── __init__.py           # New tool schemas (UPDATED)
├── train_model.py            # --with-text flag (UPDATED)
└── http_server.py            # New endpoint (UPDATED)

server/
├── src/
│   ├── planner/
│   │   └── executePlan.ts    # Tool routing (UPDATED)
│   └── services/
│       └── mcpClient.ts      # New tool caller (UPDATED)

frontend/
├── src/
│   ├── lib/
│   │   └── api.ts            # New types (UPDATED)
│   └── components/
│       └── ToolResultDisplay.tsx  # Sentiment display (UPDATED)
```

## Phase 4 Design Decisions

### Deterministic Sentiment
- FinBERT uses pre-trained model (no fine-tuning)
- Mock mode uses keyword matching for testing
- Same text always returns same sentiment score

### Dual Model Architecture
- Price-only model unchanged (backward compatible)
- Price+text model adds 2 features
- Planner routes based on `use_text_features` flag

### No Dynamic Model Switching
- Model is loaded once at server startup
- Model type is fixed by environment config
- No runtime model selection

## Troubleshooting Phase 4

### FinBERT Not Loading
```bash
# Install transformers
pip install transformers torch

# Or use mock mode
export NEWS_MODEL_TYPE=mock
```

### Price+Text Model Not Found
```bash
# Train the model
cd mcp_server
python train_model.py --with-text
```

### Sentiment Score Always 0
```bash
# Check if news_text is being passed
# Ensure use_text_features=true in plan
# Check NEWS_MODEL_TYPE in .env
```

---

**Phase 4 Rule:** Sentiment is computed, not interpreted. The model learns from features, not from news understanding.
