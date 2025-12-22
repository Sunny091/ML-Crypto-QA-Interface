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
            chart_data: 圖表數據（如果是圖表請求）
        """
        from orchestrator.agent import CryptoAgent
        import re

        try:
            data = request.get_json() or {}
            message = data.get("message", "").strip()

            if not message:
                return jsonify({"success": False, "error": "請輸入訊息"})

            # 檢測幣種
            symbol = None
            for s in ["btc", "eth", "sol"]:
                if s in message.lower():
                    symbol = s.upper()
                    break

            # 檢測是否為「預測」請求（預測未來漲跌，不是圖表）
            predict_keywords = ["預測", "predict", "明天", "會漲", "會跌", "漲跌"]
            is_predict_request = any(kw in message.lower() for kw in predict_keywords)

            # 如果是預測請求，交給 LLM Agent 處理
            if is_predict_request:
                pass  # 跳過圖表處理，讓後面的 LLM Agent 處理

            # 檢測是否涉及時間範圍（用於判斷是否需要圖表）
            has_time_range = (
                re.search(r'\d{1,2}[月\/]\d{1,2}', message) or  # 日期格式
                re.search(r'\d{4}-\d{1,2}-\d{1,2}', message) or  # YYYY-MM-DD
                re.search(r'(?:近|過去|最近|從)\s*\d+\s*(?:天|日|週|周|月)', message.lower()) or
                any(kw in message.lower() for kw in ["到現在", "至今", "到今天", "一週", "一周", "一個月", "三個月"])
            )

            # 明確的圖表關鍵詞
            chart_keywords = ["走勢", "圖表", "chart", "歷史", "價格圖", "趨勢圖"]
            explicit_chart_request = any(kw in message.lower() for kw in chart_keywords)

            # 如果有幣種且（明確要求圖表 或 涉及時間範圍）且不是預測請求，則解析時間
            if symbol and (explicit_chart_request or has_time_range) and not is_predict_request:
                from data.scrapers.coincap_client import get_price_history
                from datetime import datetime

                today = datetime.now().strftime("%Y-%m-%d")
                chart_params = {"symbol": symbol}

                # 解析「到現在」格式
                to_now_match = re.search(
                    r'(?:從)?(\d{1,2}月\d{1,2}日?|\d{1,2}[\/]\d{1,2}|\d{4}-\d{1,2}-\d{1,2})\s*(?:到|至)?\s*(?:現在|今天|今日|now|today|至今)',
                    message, re.IGNORECASE
                )

                # 解析日期範圍格式
                date_range_match = re.search(
                    r'(\d{1,2}月\d{1,2}日?|\d{1,2}[\/]\d{1,2}|\d{4}-\d{1,2}-\d{1,2})\s*(?:到|至|~)\s*(\d{1,2}月\d{1,2}日?|\d{1,2}[\/]\d{1,2}|\d{4}-\d{1,2}-\d{1,2})',
                    message
                )

                # 解析天數格式
                days_match = re.search(r'(?:近|過去|最近)?\s*(\d+)\s*(?:天|日|days?)', message.lower())

                def parse_date(date_str):
                    year = datetime.now().year
                    if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):
                        parts = date_str.split('-')
                        return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                    if re.match(r'^\d{1,2}/\d{1,2}$', date_str):
                        parts = date_str.split('/')
                        return f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                    cn_match = re.match(r'(\d{1,2})月(\d{1,2})日?', date_str)
                    if cn_match:
                        return f"{year}-{cn_match.group(1).zfill(2)}-{cn_match.group(2).zfill(2)}"
                    return None

                if to_now_match:
                    start_date = parse_date(to_now_match.group(1))
                    if start_date:
                        chart_params["start_date"] = start_date
                        chart_params["end_date"] = today
                elif date_range_match:
                    start_date = parse_date(date_range_match.group(1))
                    end_date = parse_date(date_range_match.group(2))
                    if start_date and end_date:
                        chart_params["start_date"] = start_date
                        chart_params["end_date"] = end_date
                elif days_match:
                    chart_params["days"] = min(int(days_match.group(1)), 365)
                else:
                    # 匹配週/月的中文表達
                    msg_lower = message.lower()
                    if "一週" in msg_lower or "一周" in msg_lower:
                        chart_params["days"] = 7
                    elif "兩週" in msg_lower or "两周" in msg_lower:
                        chart_params["days"] = 14
                    elif "一個月" in msg_lower or "一个月" in msg_lower:
                        chart_params["days"] = 30
                    elif "三個月" in msg_lower or "三个月" in msg_lower:
                        chart_params["days"] = 90
                    else:
                        chart_params["days"] = 30  # 預設 30 天

                # 獲取價格數據
                history = get_price_history(
                    symbol,
                    days=chart_params.get("days"),
                    start_date=chart_params.get("start_date"),
                    end_date=chart_params.get("end_date")
                )

                # 計算時間跨度（天數）
                if chart_params.get("start_date") and chart_params.get("end_date"):
                    from datetime import datetime as dt
                    start_dt = dt.strptime(chart_params["start_date"], "%Y-%m-%d")
                    end_dt = dt.strptime(chart_params["end_date"], "%Y-%m-%d")
                    time_span_days = (end_dt - start_dt).days + 1
                    period_label = f"{chart_params['start_date']} ~ {chart_params['end_date']}"
                else:
                    time_span_days = chart_params.get("days", 30)
                    period_label = f"近 {time_span_days} 天"

                # 決定使用圖表還是文字
                # 明確要求圖表 或 時間跨度 > 3天 → 使用圖表
                # 否則使用文字描述
                use_chart = explicit_chart_request or time_span_days > 3

                if use_chart:
                    return jsonify({
                        "success": True,
                        "response": f"以下是 {symbol} {period_label} 的價格走勢圖：",
                        "chart_data": {
                            "symbol": symbol,
                            "timestamps": [p["timestamp"] for p in history["data"]],
                            "prices": [p["price_usd"] for p in history["data"]],
                            "period": period_label,
                            "source": history["source"]
                        }
                    })
                else:
                    # 短時間範圍，使用文字描述
                    data_points = history["data"]
                    if len(data_points) >= 2:
                        first_price = float(data_points[0]["price_usd"])
                        last_price = float(data_points[-1]["price_usd"])
                        change = last_price - first_price
                        change_pct = (change / first_price) * 100
                        trend = "上漲 📈" if change > 0 else "下跌 📉" if change < 0 else "持平"

                        response_text = f"""**{symbol} {period_label} 價格變化：**

• 起始價格：${first_price:,.2f}
• 最新價格：${last_price:,.2f}
• 變化：{'+' if change > 0 else ''}{change:,.2f} ({'+' if change_pct > 0 else ''}{change_pct:.2f}%)
• 趨勢：{trend}

資料來源：{history['source']}"""
                    else:
                        response_text = f"無法獲取 {symbol} {period_label} 的價格數據。"

                    return jsonify({
                        "success": True,
                        "response": response_text
                    })

            # 非圖表請求，使用 LLM 處理
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
