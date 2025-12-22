"""
LLM Agent - 大腦層

使用 LLM (Ollama) 作為大腦，理解用戶自然語言並調用 MCP 工具。
實現 ReAct 模式：Reasoning + Acting
"""

import sys
import json
import re
from pathlib import Path
from typing import Generator
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.tools import get_available_tools, execute_tool, get_tools_for_llm


class CryptoAgent:
    """
    加密貨幣預測 Agent

    使用 LLM 理解用戶意圖，透過 MCP 工具執行操作。
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0,
        api_key: str = ""
    ):
        self.ollama_host = ollama_host.rstrip('/')  # 移除尾部斜線
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.tools = get_available_tools()

        # 系統提示詞
        self.system_prompt = """你是一個加密貨幣分析助手。你可以使用以下工具來幫助用戶：

可用工具：
1. get_current_price - 獲取即時價格（支援 BTC, ETH, SOL）
2. get_price_history - 獲取歷史價格數據
3. predict_price - 使用 ML 模型預測明天漲跌
4. predict_with_news - 結合新聞情感預測
5. analyze_sentiment - 分析新聞情感
6. get_technical_analysis - 獲取技術分析指標

當用戶詢問時，你需要：
1. 理解用戶意圖
2. 選擇合適的工具
3. 解讀工具返回的結果
4. 用自然語言回答用戶

回答時請：
- 使用繁體中文
- 數據要清楚標示
- 預測結果要說明信心程度
- 適當加入分析解讀"""

    def _call_ollama(self, messages: list, use_tools: bool = True) -> dict:
        """調用 Ollama API"""
        url = f"{self.ollama_host}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        # 添加工具定義
        if use_tools:
            payload["tools"] = get_tools_for_llm()

        data = json.dumps(payload).encode('utf-8')

        # 設定 headers（包含 API Key）
        headers = {
            'Content-Type': 'application/json',
            'accept': 'application/json'
        }
        if self.api_key:
            headers['X-API-Key'] = self.api_key

        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise ConnectionError(f"無法連接 Ollama: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama 調用失敗: {e}")

    def _call_ollama_simple(self, prompt: str) -> str:
        """簡單調用 Ollama（不使用工具，使用 /api/chat）"""
        url = f"{self.ollama_host}/api/chat"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        data = json.dumps(payload).encode('utf-8')

        # 設定 headers（包含 API Key）
        headers = {
            'Content-Type': 'application/json',
            'accept': 'application/json'
        }
        if self.api_key:
            headers['X-API-Key'] = self.api_key

        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("message", {}).get("content", "")
        except Exception as e:
            return f"LLM 調用失敗: {e}"

    def _parse_tool_calls(self, response: dict) -> list:
        """解析 LLM 回應中的工具調用"""
        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])
        return tool_calls

    def _execute_tools(self, tool_calls: list) -> list:
        """執行工具調用"""
        results = []
        for call in tool_calls:
            func = call.get("function", {})
            tool_name = func.get("name")
            arguments = func.get("arguments", {})

            print(f"[Agent] 調用工具: {tool_name}({arguments})")
            result = execute_tool(tool_name, arguments)
            results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

        return results

    def _parse_date(self, date_str: str) -> str:
        """解析日期字串為 YYYY-MM-DD 格式"""
        from datetime import datetime
        now = datetime.now()
        year = now.year

        # 格式: YYYY-MM-DD
        if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):
            parts = date_str.split('-')
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"

        # 格式: MM/DD 或 M/D
        if re.match(r'^\d{1,2}/\d{1,2}$', date_str):
            parts = date_str.split('/')
            return f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

        # 格式: MM-DD 或 M-D
        if re.match(r'^\d{1,2}-\d{1,2}$', date_str):
            parts = date_str.split('-')
            return f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

        # 格式: X月X日
        cn_match = re.match(r'(\d{1,2})月(\d{1,2})日?', date_str)
        if cn_match:
            return f"{year}-{cn_match.group(1).zfill(2)}-{cn_match.group(2).zfill(2)}"

        return None

    def _fallback_tool_detection(self, user_message: str) -> list:
        """備用工具檢測（當 LLM 不支援 function calling 時）"""
        message_lower = user_message.lower()
        tool_calls = []

        # 檢測價格查詢
        for symbol in ["btc", "eth", "sol"]:
            if symbol in message_lower:
                symbol_upper = symbol.upper()

                if any(word in message_lower for word in ["價格", "多少錢", "現在", "即時", "price"]):
                    tool_calls.append({
                        "tool": "get_current_price",
                        "arguments": {"symbol": symbol_upper}
                    })

                if any(word in message_lower for word in ["預測", "明天", "走勢", "漲", "跌", "predict"]):
                    tool_calls.append({
                        "tool": "predict_price",
                        "arguments": {"symbol": symbol_upper}
                    })

                if any(word in message_lower for word in ["技術", "分析", "rsi", "指標", "technical"]):
                    tool_calls.append({
                        "tool": "get_technical_analysis",
                        "arguments": {"symbol": symbol_upper}
                    })

                # 歷史價格 - 支援多種時間格式
                # 當有日期範圍或天數時，也觸發歷史價格查詢
                has_date_indicator = (
                    any(word in message_lower for word in ["歷史", "過去", "history", "走勢", "圖表", "chart", "近", "最近", "從", "到現在", "至今", "到今天"]) or
                    re.search(r'\d{1,2}[月\/]\d{1,2}', user_message) or  # 日期格式 (X月X日, X/X)
                    re.search(r'\d{4}-\d{1,2}-\d{1,2}', user_message) or  # 日期格式 (YYYY-MM-DD)
                    re.search(r'(\d+)\s*(?:天|日|週|周|月)', message_lower)  # 天數格式
                )
                if has_date_indicator:
                    from datetime import datetime
                    arguments = {"symbol": symbol_upper}
                    today = datetime.now().strftime("%Y-%m-%d")

                    # 嘗試匹配「從 X 到現在」: "從12/1到現在", "12月1日至今", "從2024-12-01到今天"
                    to_now_match = re.search(
                        r'(?:從)?(\d{1,2}月\d{1,2}日?|\d{1,2}[\/]\d{1,2}|\d{4}-\d{1,2}-\d{1,2})\s*(?:到|至)?\s*(?:現在|今天|今日|now|today|至今)',
                        user_message, re.IGNORECASE
                    )
                    if to_now_match:
                        start_date = self._parse_date(to_now_match.group(1))
                        if start_date:
                            arguments["start_date"] = start_date
                            arguments["end_date"] = today

                    # 嘗試匹配日期範圍: "12/1到12/15", "12月1日到12月15日", "2024-12-01到2024-12-15"
                    elif not to_now_match:
                        date_range_match = re.search(
                            r'(\d{1,2}月\d{1,2}日?|\d{1,2}[\/]\d{1,2}|\d{4}-\d{1,2}-\d{1,2})\s*(?:到|至|~)\s*(\d{1,2}月\d{1,2}日?|\d{1,2}[\/]\d{1,2}|\d{4}-\d{1,2}-\d{1,2})',
                            user_message
                        )
                        if date_range_match:
                            start_date = self._parse_date(date_range_match.group(1))
                            end_date = self._parse_date(date_range_match.group(2))
                            if start_date and end_date:
                                arguments["start_date"] = start_date
                                arguments["end_date"] = end_date

                    if "start_date" not in arguments:
                        # 嘗試匹配天數: "近7天", "過去30天", "最近一週"
                        days = 30  # 預設值
                        days_match = re.search(r'(?:近|過去|最近)\s*(\d+)\s*(?:天|日|days?)', message_lower)
                        if days_match:
                            days = min(int(days_match.group(1)), 365)
                        else:
                            # 匹配 "X天" 或 "X日"
                            simple_days_match = re.search(r'(\d+)\s*(?:天|日|days?)', message_lower)
                            if simple_days_match:
                                days = min(int(simple_days_match.group(1)), 365)
                            # 匹配週/月
                            elif "一週" in message_lower or "一周" in message_lower:
                                days = 7
                            elif "兩週" in message_lower or "两周" in message_lower:
                                days = 14
                            elif "一個月" in message_lower or "一个月" in message_lower:
                                days = 30
                            elif "三個月" in message_lower or "三个月" in message_lower:
                                days = 90
                        arguments["days"] = days

                    tool_calls.append({
                        "tool": "get_price_history",
                        "arguments": arguments
                    })

                break  # 只處理第一個匹配的幣種

        return tool_calls

    def chat(self, user_message: str) -> str:
        """
        處理用戶訊息

        Args:
            user_message: 用戶輸入

        Returns:
            Agent 回應
        """
        print(f"[Agent] 收到訊息: {user_message}")

        # 嘗試使用 function calling
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = self._call_ollama(messages, use_tools=True)
            tool_calls = self._parse_tool_calls(response)

            # 如果 LLM 返回了工具調用
            if tool_calls:
                tool_results = self._execute_tools([
                    {"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in tool_calls
                ])

                # 讓 LLM 解讀結果
                result_text = json.dumps(tool_results, ensure_ascii=False, indent=2)
                interpretation_prompt = f"""用戶問：{user_message}

工具執行結果：
{result_text}

請用繁體中文自然地回答用戶的問題，解讀上述數據。"""

                return self._call_ollama_simple(interpretation_prompt)

            # LLM 直接回答（沒有調用工具）
            message = response.get("message", {})
            content = message.get("content", "")
            if content:
                return content

        except Exception as e:
            print(f"[Agent] Function calling 失敗: {e}")

        # 備用方案：手動檢測工具
        print("[Agent] 使用備用工具檢測")
        fallback_calls = self._fallback_tool_detection(user_message)

        if fallback_calls:
            tool_results = []
            for call in fallback_calls:
                result = execute_tool(call["tool"], call["arguments"])
                tool_results.append({
                    "tool": call["tool"],
                    "result": result
                })

            # 讓 LLM 解讀結果
            result_text = json.dumps(tool_results, ensure_ascii=False, indent=2)
            interpretation_prompt = f"""用戶問：{user_message}

工具執行結果：
{result_text}

請用繁體中文自然地回答用戶的問題。如果是價格，請標明幣種和金額。如果是預測，請說明預測結果和信心程度。"""

            return self._call_ollama_simple(interpretation_prompt)

        # 沒有工具調用，直接回答
        return self._call_ollama_simple(f"{self.system_prompt}\n\n用戶：{user_message}\n\n請用繁體中文回答：")

    def check_connection(self) -> dict:
        """檢查 Ollama 連接狀態"""
        try:
            url = f"{self.ollama_host}/api/tags"
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "status": "connected",
                    "host": self.ollama_host,
                    "models": models,
                    "current_model": self.model
                }
        except Exception as e:
            return {
                "status": "disconnected",
                "error": str(e)
            }


# 測試
if __name__ == "__main__":
    agent = CryptoAgent()

    # 檢查連接
    status = agent.check_connection()
    print(f"Ollama 狀態: {status}")

    if status["status"] == "connected":
        # 測試對話
        response = agent.chat("BTC 現在多少錢？")
        print(f"\n回應: {response}")
