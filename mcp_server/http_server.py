#!/usr/bin/env python3
"""
HTTP API Server for MCP tools.

Phase 3: Provides HTTP interface for Node.js backend to call tools.
This wraps the same tool logic used by fastmcp, but exposes it via HTTP.

Usage:
    python http_server.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from http.server import HTTPServer, BaseHTTPRequestHandler
from config import env
from data import get_price_data
from features import get_latest_features, get_price_text_features, PRICE_ONLY_FEATURES, PRICE_TEXT_FEATURES
from models import get_predictor, get_text_predictor
from news import get_news_features
from tools import (
    PredictPriceInput,
    PredictPriceOutput,
    PredictPriceWithTextInput,
    PredictPriceWithTextOutput,
)


class ToolHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MCP tools."""

    def _send_json_response(self, status: int, data: dict):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self._send_json_response(200, {
                'status': 'ok',
                'server': env.mcp.server_name,
                'tools': ['predict_price_price_only', 'predict_price_with_text'],
            })
        elif self.path == '/tools':
            self._send_json_response(200, {
                'tools': [
                    {
                        'name': 'predict_price_price_only',
                        'description': 'Predict next-day price movement using price data only',
                        'input_schema': {
                            'symbol': 'BTC | ETH | SOL',
                            'as_of': 'YYYY-MM-DD',
                        },
                    },
                    {
                        'name': 'predict_price_with_text',
                        'description': 'Predict next-day price movement using price data and news sentiment',
                        'input_schema': {
                            'symbol': 'BTC | ETH | SOL',
                            'as_of': 'YYYY-MM-DD',
                            'news_text': 'string',
                        },
                    }
                ]
            })
        else:
            self._send_json_response(404, {'error': 'Not found'})

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/tools/predict_price_price_only':
            self._handle_predict_price()
        elif self.path == '/tools/predict_price_with_text':
            self._handle_predict_price_with_text()
        else:
            self._send_json_response(404, {'error': f'Tool not found: {self.path}'})

    def _handle_predict_price(self):
        """Handle predict_price_price_only tool call."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            data = json.loads(body) if body else {}

            print(f"[HTTP] predict_price_price_only called")
            print(f"  - Input: {data}")

            # Validate input
            try:
                input_data = PredictPriceInput(**data)
            except Exception as e:
                self._send_json_response(400, {
                    'error': 'Invalid input',
                    'details': str(e),
                })
                return

            # Resolve "now" to actual date
            if input_data.as_of == "now":
                as_of_date = datetime.now().strftime("%Y-%m-%d")
            else:
                as_of_date = input_data.as_of

            # Get price data
            df = get_price_data(
                symbol=input_data.symbol,
                as_of=as_of_date,
                lookback_days=60,
            )

            # Build features
            features = get_latest_features(df)
            print(f"  - Features: {features}")

            # Get prediction
            predictor = get_predictor()
            prediction, prob_up = predictor.predict(features)

            print(f"  - Prediction: {prediction} (prob_up={prob_up:.4f})")

            # Build response
            result = PredictPriceOutput(
                symbol=input_data.symbol,
                as_of=as_of_date,
                prediction=prediction,
                prob_up=round(prob_up, 4),
                model_variant="price_only",
                features_used=PRICE_ONLY_FEATURES,
            )

            self._send_json_response(200, result.model_dump())

        except Exception as e:
            print(f"[HTTP] Error: {e}")
            self._send_json_response(500, {
                'error': 'Internal server error',
                'details': str(e),
            })

    def _handle_predict_price_with_text(self):
        """Handle predict_price_with_text tool call (Phase 4)."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            data = json.loads(body) if body else {}

            print(f"[HTTP] predict_price_with_text called")
            print(f"  - Input: symbol={data.get('symbol')}, as_of={data.get('as_of')}, news_text={data.get('news_text', '')[:50]}...")

            # Validate input
            try:
                input_data = PredictPriceWithTextInput(**data)
            except Exception as e:
                self._send_json_response(400, {
                    'error': 'Invalid input',
                    'details': str(e),
                })
                return

            # Resolve "now" to actual date
            if input_data.as_of == "now":
                as_of_date = datetime.now().strftime("%Y-%m-%d")
            else:
                as_of_date = input_data.as_of

            # Get price data
            df = get_price_data(
                symbol=input_data.symbol,
                as_of=as_of_date,
                lookback_days=60,
            )

            # Build combined features (price + text)
            features = get_price_text_features(df, input_data.news_text)
            print(f"  - Features: {features}")

            # Get prediction from price+text model
            predictor = get_text_predictor()
            prediction, prob_up = predictor.predict(features)

            print(f"  - Prediction: {prediction} (prob_up={prob_up:.4f})")

            # Build response
            result = PredictPriceWithTextOutput(
                symbol=input_data.symbol,
                as_of=as_of_date,
                prediction=prediction,
                prob_up=round(prob_up, 4),
                model_variant="price_plus_text",
                features_used=PRICE_TEXT_FEATURES,
                sentiment_score=features["sentiment_score"],
            )

            self._send_json_response(200, result.model_dump())

        except Exception as e:
            print(f"[HTTP] Error: {e}")
            self._send_json_response(500, {
                'error': 'Internal server error',
                'details': str(e),
            })

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[HTTP] {self.address_string()} - {format % args}")


def main():
    """Start the HTTP server."""
    print("")
    print("=" * 60)
    print("MCP Tool HTTP Server - Phase 4")
    print("=" * 60)
    print(f"  Server Name:   {env.mcp.server_name}")
    print(f"  Host:          {env.mcp.server_host}")
    print(f"  Port:          {env.mcp.server_port}")
    print(f"  Environment:   {env.app.app_env}")
    print(f"  News Model:    {env.news.model_type}")
    print("=" * 60)
    print("")

    # Pre-load models
    print("[Startup] Loading price-only model...")
    try:
        predictor = get_predictor()
        print("[Startup] Price-only model loaded successfully!")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] Run 'python train_model.py' first to create the model.")
        sys.exit(1)

    print("[Startup] Loading price+text model...")
    try:
        text_predictor = get_text_predictor()
        print("[Startup] Price+text model loaded successfully!")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] Run 'python train_model.py --with-text' to create the model.")
        print("[Startup] Continuing without price+text model...")

    print("")
    print("[Server] Starting HTTP server...")
    print("[Server] Endpoints:")
    print(f"  - GET  http://{env.mcp.server_host}:{env.mcp.server_port}/health")
    print(f"  - GET  http://{env.mcp.server_host}:{env.mcp.server_port}/tools")
    print(f"  - POST http://{env.mcp.server_host}:{env.mcp.server_port}/tools/predict_price_price_only")
    print(f"  - POST http://{env.mcp.server_host}:{env.mcp.server_port}/tools/predict_price_with_text")
    print("")

    server = HTTPServer((env.mcp.server_host, env.mcp.server_port), ToolHandler)

    try:
        print(f"[Server] Listening on http://{env.mcp.server_host}:{env.mcp.server_port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
