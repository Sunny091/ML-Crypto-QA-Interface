#!/bin/bash

# Crypto Predictor MCP - 啟動腳本

cd "$(dirname "$0")"

echo "=========================================="
echo "  Crypto Predictor MCP Server"
echo "=========================================="

# 檢查 Node.js
if ! command -v node &> /dev/null; then
    echo "錯誤: 找不到 Node.js"
    exit 1
fi

# 檢查 Python
if ! command -v python &> /dev/null; then
    echo "錯誤: 找不到 Python"
    exit 1
fi

# 安裝 Node.js 依賴
if [ ! -d "node_modules" ]; then
    echo "[Setup] 安裝 Node.js 依賴..."
    npm install
fi

# 載入環境變數
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

PORT=${PORT:-3000}
HOST=${HOST:-0.0.0.0}

echo ""
echo "  URL: http://${HOST}:${PORT}"
echo "  API: http://${HOST}:${PORT}/api"
echo "=========================================="
echo ""

# 啟動服務
npm start
