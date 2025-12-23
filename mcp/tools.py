"""
MCP Tools - 工具定義層

定義所有可供 LLM 調用的工具。
這些工具會被 Orchestrator 透過 function calling 調用。
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Literal, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== 工具定義 =====

TOOLS = [
    {
        "name": "get_current_price",
        "description": "獲取加密貨幣的即時價格。返回價格、24小時漲跌幅、成交量等資訊。",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ["BTC", "ETH", "SOL"],
                    "description": "加密貨幣符號"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_price_history",
        "description": "獲取加密貨幣的歷史價格數據。可以用 days 指定近幾天，或用 start_date/end_date 指定日期範圍。例如：近7天用 days=7，指定範圍用 start_date='2024-12-01' 和 end_date='2024-12-15'。",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ["BTC", "ETH", "SOL"],
                    "description": "加密貨幣符號"
                },
                "days": {
                    "type": "integer",
                    "description": "查詢近幾天的數據 (1-365)，與 start_date/end_date 二選一"
                },
                "start_date": {
                    "type": "string",
                    "description": "開始日期，格式 YYYY-MM-DD，例如 2024-12-01"
                },
                "end_date": {
                    "type": "string",
                    "description": "結束日期，格式 YYYY-MM-DD，例如 2024-12-15"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "predict_price",
        "description": "使用深度學習模型預測加密貨幣明天的價格走勢（漲或跌）。可選擇 Transformer、LSTM 或 LightGBM 模型。預設使用環境配置中的模型。",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ["BTC", "ETH", "SOL"],
                    "description": "加密貨幣符號"
                },
                "model": {
                    "type": "string",
                    "enum": ["transformer", "lstm", "lightgbm"],
                    "description": "使用的模型類型（預設為環境配置中的 PREDICTION_MODEL）"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "predict_with_news",
        "description": "結合新聞情感分析預測價格走勢。需要提供新聞文本。",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ["BTC", "ETH", "SOL"],
                    "description": "加密貨幣符號"
                },
                "news_text": {
                    "type": "string",
                    "description": "新聞文本內容"
                }
            },
            "required": ["symbol", "news_text"]
        }
    },
    {
        "name": "analyze_sentiment",
        "description": "分析新聞文本的市場情感（看漲/看跌/中性）。",
        "parameters": {
            "type": "object",
            "properties": {
                "news_text": {
                    "type": "string",
                    "description": "要分析的新聞文本"
                }
            },
            "required": ["news_text"]
        }
    },
    {
        "name": "get_technical_analysis",
        "description": "獲取技術分析指標，包括 RSI、波動率、移動平均線等。",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ["BTC", "ETH", "SOL"],
                    "description": "加密貨幣符號"
                }
            },
            "required": ["symbol"]
        }
    },
]


def get_available_tools() -> list[dict]:
    """獲取所有可用工具的定義"""
    return TOOLS


def get_tools_for_llm() -> list[dict]:
    """獲取 LLM function calling 格式的工具定義"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
        for tool in TOOLS
    ]


# ===== 工具實現 =====

def _get_current_price(symbol: str) -> dict:
    """獲取即時價格"""
    from data.scrapers.coincap_client import get_current_price
    return get_current_price(symbol)


def _get_price_history(symbol: str, days: int = None, start_date: str = None, end_date: str = None) -> dict:
    """獲取歷史價格"""
    from data.scrapers.coincap_client import get_price_history
    return get_price_history(symbol, days=days, start_date=start_date, end_date=end_date)


def _predict_price(symbol: str, model: str = None) -> dict:
    """ML 模型預測"""
    from configs.config import env

    # 如果沒有指定模型，使用環境配置
    if model is None:
        model = env.model.prediction_model

    # 統一轉成小寫
    model = model.lower()

    # 使用 Transformer 模型
    if model == "transformer":
        try:
            from models.transformer.predictor import predict_with_transformer
            result = predict_with_transformer(symbol)
            if "error" not in result:
                confidence = result.get("confidence", 0.5)
                result["confidence_level"] = "高" if confidence > 0.7 else "中" if confidence > 0.55 else "低"
                return result
        except Exception as e:
            print(f"[MCP] Transformer 預測失敗: {e}")
            return {"error": f"Transformer 模型預測失敗: {e}"}

    # 使用 LSTM 模型
    if model == "lstm":
        try:
            from models.lstm.predictor import predict_with_lstm
            result = predict_with_lstm(symbol)
            if "error" not in result:
                confidence = result.get("confidence", 0.5)
                result["confidence_level"] = "高" if confidence > 0.7 else "中" if confidence > 0.55 else "低"
                return result
        except Exception as e:
            print(f"[MCP] LSTM 預測失敗: {e}")
            return {"error": f"LSTM 模型預測失敗: {e}"}

    # 使用 LightGBM 模型
    if model == "lightgbm":
        from models.predictor import get_predictor
        from etl.extract import extract_price_data
        from etl.transform import get_latest_features

        try:
            df = extract_price_data(symbol, "now", lookback_days=60)
            features = get_latest_features(df)
            predictor = get_predictor()
            prediction, prob_up = predictor.predict(features)

            return {
                "symbol": symbol,
                "prediction": prediction,
                "probability": round(prob_up, 4),
                "confidence": round(prob_up if prediction == "UP" else 1 - prob_up, 4),
                "confidence_level": "高" if abs(prob_up - 0.5) > 0.2 else "中" if abs(prob_up - 0.5) > 0.1 else "低",
                "features": features,
                "model": "LightGBM",
                "timestamp": datetime.now().isoformat()
            }
        except FileNotFoundError:
            return {"error": "LightGBM 模型尚未訓練"}

    return {"error": f"未知的模型類型: {model}"}


def _predict_with_news(symbol: str, news_text: str) -> dict:
    """結合新聞的預測"""
    from models.predictor import get_text_predictor
    from etl.extract import extract_price_data
    from etl.transform import get_price_text_features, PRICE_TEXT_FEATURES

    try:
        df = extract_price_data(symbol, "now", lookback_days=60)
        features = get_price_text_features(df, news_text)

        predictor = get_text_predictor()
        prediction, prob_up = predictor.predict(features)

        return {
            "symbol": symbol,
            "prediction": prediction,
            "probability": round(prob_up, 4),
            "sentiment_score": features["sentiment_score"],
            "model": "LightGBM + Sentiment",
            "timestamp": datetime.now().isoformat()
        }
    except FileNotFoundError:
        return {
            "error": "模型尚未訓練，請先執行 python -m models.training.train --with-text"
        }


def _analyze_sentiment(news_text: str) -> dict:
    """情感分析"""
    from models.sentiment import analyze_news_full
    return analyze_news_full(news_text)


def _get_technical_analysis(symbol: str) -> dict:
    """技術分析"""
    from etl.extract import extract_price_data
    from etl.transform import build_features

    df = extract_price_data(symbol, "now", lookback_days=60)
    df = build_features(df)

    latest = df.iloc[-1]

    # RSI 解讀
    rsi = latest["rsi_14"]
    if rsi > 70:
        rsi_signal = "超買（可能回調）"
    elif rsi < 30:
        rsi_signal = "超賣（可能反彈）"
    else:
        rsi_signal = "中性"

    return {
        "symbol": symbol,
        "indicators": {
            "rsi_14": round(rsi, 2),
            "rsi_signal": rsi_signal,
            "return_1d": f"{latest['return_1d']*100:.2f}%",
            "volatility_7d": f"{latest['volatility_7d']*100:.2f}%",
            "sma_20": round(latest["sma_20"], 2),
            "sma_50": round(latest["sma_50"], 2),
            "current_price": round(latest["close"], 2),
        },
        "trend": "上升" if latest["close"] > latest["sma_20"] else "下降",
        "timestamp": datetime.now().isoformat()
    }


# ===== 工具執行器 =====

def execute_tool(tool_name: str, arguments: dict) -> dict:
    """
    執行指定的工具

    Args:
        tool_name: 工具名稱
        arguments: 工具參數

    Returns:
        工具執行結果
    """
    tool_map = {
        "get_current_price": _get_current_price,
        "get_price_history": _get_price_history,
        "predict_price": _predict_price,
        "predict_with_news": _predict_with_news,
        "analyze_sentiment": _analyze_sentiment,
        "get_technical_analysis": _get_technical_analysis,
    }

    if tool_name not in tool_map:
        return {"error": f"未知工具: {tool_name}"}

    try:
        result = tool_map[tool_name](**arguments)
        return result
    except Exception as e:
        return {"error": str(e)}
