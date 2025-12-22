"""
API Module - Flask API 部署
提供 RESTful API 服務
"""
from api.app import create_app, app

__all__ = ["create_app", "app"]
