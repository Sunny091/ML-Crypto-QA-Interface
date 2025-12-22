#!/bin/bash

# MCP Orchestrator 啟動腳本 (重定向到 mcp_server/)

cd "$(dirname "$0")/mcp_server"
./start.sh
