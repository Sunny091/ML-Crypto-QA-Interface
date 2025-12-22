"""
Flask API Application - 整合 LLM + MCP 的完整服務

架構：
- Flask API 作為對外服務（滿足報告要求）
- LLM (Ollama) 作為大腦理解用戶自然語言
- MCP Tools 作為工具層執行實際操作
- 核心模組 (data/, etl/, models/) 提供 ML 功能

Usage:
    python -m api.app
"""

import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import env


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder='static')

    # ===== 靜態文件和前端 =====

    @app.route("/")
    def index():
        """Serve the frontend"""
        return send_from_directory(app.static_folder, 'index.html')

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        from mcp.tools import get_available_tools
        return jsonify({
            "status": "ok",
            "tools": len(get_available_tools()),
            "timestamp": datetime.now().isoformat(),
        })

    # ===== LLM Chat Endpoint (核心功能) =====

    @app.route("/api/chat", methods=["POST"])
    def chat():
        """
        LLM 聊天端點 - 用戶用自然語言與系統互動

        POST JSON:
            message: 用戶訊息

        Returns:
            success: bool
            response: LLM 回應
        """
        from orchestrator.agent import CryptoAgent

        try:
            data = request.get_json() or {}
            message = data.get("message", "").strip()

            if not message:
                return jsonify({"success": False, "error": "請輸入訊息"})

            # 創建 Agent 並處理訊息（從 config 讀取設定）
            agent = CryptoAgent(
                ollama_host=env.ollama.host,
                model=env.ollama.model,
                timeout=env.ollama.timeout_ms / 1000.0,
                api_key=env.ollama.api_key
            )

            response = agent.chat(message)

            return jsonify({
                "success": True,
                "response": response
            })

        except ConnectionError as e:
            return jsonify({
                "success": False,
                "error": f"無法連接 LLM 服務: {e}",
                "hint": "請確保 Ollama 正在運行"
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            })

    # ===== 圖表數據端點 =====

    @app.route("/api/chart-data", methods=["POST"])
    def get_chart_data():
        """獲取圖表數據"""
        from data.scrapers.coincap_client import get_price_history

        try:
            data = request.get_json() or {}
            symbol = data.get("symbol", "BTC").upper()
            days = data.get("days", 30)
            start_date = data.get("start_date")
            end_date = data.get("end_date")

            if symbol not in ["BTC", "ETH", "SOL"]:
                return jsonify({"error": f"不支援的幣種: {symbol}"}), 400

            history = get_price_history(
                symbol,
                days=days,
                start_date=start_date,
                end_date=end_date
            )

            # 轉換為前端需要的格式
            timestamps = [p["timestamp"] for p in history["data"]]
            prices = [p["price_usd"] for p in history["data"]]

            return jsonify({
                "symbol": symbol,
                "timestamps": timestamps,
                "prices": prices,
                "source": history["source"]
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ===== 直接 API 端點（不透過 LLM）=====

    @app.route("/api/price/<symbol>", methods=["GET"])
    def get_current_price(symbol: str):
        """獲取即時價格"""
        from data.scrapers.coincap_client import get_current_price as fetch_price

        try:
            symbol = symbol.upper()
            if symbol not in ["BTC", "ETH", "SOL"]:
                return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
            return jsonify(fetch_price(symbol))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/predict/<symbol>", methods=["GET"])
    def predict_price(symbol: str):
        """ML 模型預測"""
        from mcp.tools import execute_tool

        try:
            symbol = symbol.upper()
            if symbol not in ["BTC", "ETH", "SOL"]:
                return jsonify({"error": f"Invalid symbol: {symbol}"}), 400

            result = execute_tool("predict_price", {"symbol": symbol})
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/technical/<symbol>", methods=["GET"])
    def technical_analysis(symbol: str):
        """技術分析"""
        from mcp.tools import execute_tool

        try:
            symbol = symbol.upper()
            if symbol not in ["BTC", "ETH", "SOL"]:
                return jsonify({"error": f"Invalid symbol: {symbol}"}), 400

            result = execute_tool("get_technical_analysis", {"symbol": symbol})
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/sentiment", methods=["POST"])
    def analyze_sentiment():
        """情感分析"""
        from mcp.tools import execute_tool

        try:
            data = request.get_json() or {}
            news_text = data.get("news_text", "")

            if not news_text:
                return jsonify({"error": "news_text is required"}), 400

            result = execute_tool("analyze_sentiment", {"news_text": news_text})
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ===== 系統資訊端點 =====

    @app.route("/api/tools", methods=["GET"])
    def list_tools():
        """列出所有可用工具"""
        from mcp.tools import get_available_tools
        return jsonify({"tools": get_available_tools()})

    @app.route("/api/models", methods=["GET"])
    def get_model_info():
        """獲取模型資訊"""
        models = []

        if env.model.price_only_model_path.exists():
            models.append({
                "name": "price_only",
                "status": "available",
                "description": "純價格技術指標模型"
            })
        else:
            models.append({
                "name": "price_only",
                "status": "not_trained",
                "description": "純價格技術指標模型"
            })

        if env.model.price_text_model_path.exists():
            models.append({
                "name": "price_text",
                "status": "available",
                "description": "價格 + 新聞情感模型"
            })
        else:
            models.append({
                "name": "price_text",
                "status": "not_trained",
                "description": "價格 + 新聞情感模型"
            })

        return jsonify({"models": models})

    @app.route("/api/llm/status", methods=["GET"])
    def llm_status():
        """檢查 LLM 連接狀態"""
        from orchestrator.agent import CryptoAgent

        agent = CryptoAgent(
            ollama_host=env.ollama.host,
            model=env.ollama.model,
            timeout=env.ollama.timeout_ms / 1000.0,
            api_key=env.ollama.api_key
        )
        return jsonify(agent.check_connection())

    return app


# Create the app instance
app = create_app()


def main():
    """Run the Flask development server."""
    print("=" * 60)
    print("Crypto Price Prediction System")
    print("LLM + MCP + Flask 整合架構")
    print("=" * 60)
    print(f"Host: {env.api.host}")
    print(f"Port: {env.api.port}")
    print(f"Ollama: {env.ollama.host}")
    print(f"Model: {env.ollama.model}")
    print("=" * 60)
    print(f"\n前端 UI: http://localhost:{env.api.port}")
    print("\nAPI 端點:")
    print("  POST /api/chat              - LLM 聊天 (NLP 介面)")
    print("  GET  /api/price/<symbol>    - 即時價格")
    print("  GET  /api/predict/<symbol>  - ML 預測")
    print("  GET  /api/technical/<symbol>- 技術分析")
    print("  POST /api/sentiment         - 情感分析")
    print("  GET  /api/tools             - 可用工具")
    print("  GET  /api/llm/status        - LLM 狀態")
    print("=" * 60)

    app.run(
        host=env.api.host,
        port=env.api.port,
        debug=env.api.debug,
    )


if __name__ == "__main__":
    main()
