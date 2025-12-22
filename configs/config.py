"""
Configuration module - single source of truth for all environment variables.
All env reads happen ONLY in this file.

Usage:
    from configs.config import env
"""

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class MCPConfig(BaseModel):
    """MCP Server configuration"""
    server_name: str
    server_host: str
    server_port: int

    @field_validator("server_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("server_port must be between 1 and 65535")
        return v


class ModelConfig(BaseModel):
    """Model configuration"""
    model_dir: Path
    price_only_model_name: str
    price_text_model_name: str
    transformer_model_name: str

    @property
    def price_only_model_path(self) -> Path:
        return self.model_dir / self.price_only_model_name

    @property
    def price_text_model_path(self) -> Path:
        return self.model_dir / self.price_text_model_name

    @property
    def transformer_model_path(self) -> Path:
        return self.model_dir / self.transformer_model_name


class NewsConfig(BaseModel):
    """News analysis configuration"""
    model_type: Literal["finbert", "mock"]
    max_length: int


class DataConfig(BaseModel):
    """Data source configuration"""
    price_data_source: Literal["mock", "api", "file"]
    raw_data_dir: Path
    processed_data_dir: Path


class AppConfig(BaseModel):
    """Application configuration"""
    app_env: Literal["development", "production", "test"]
    log_level: Literal["debug", "info", "warn", "error"]

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


class APIConfig(BaseModel):
    """Flask API configuration"""
    host: str
    port: int
    debug: bool


class OllamaConfig(BaseModel):
    """Ollama LLM configuration"""
    host: str
    model: str
    timeout_ms: int
    api_key: str


class Config(BaseModel):
    """Root configuration container"""
    mcp: MCPConfig
    model: ModelConfig
    data: DataConfig
    app: AppConfig
    news: NewsConfig
    api: APIConfig
    ollama: OllamaConfig


def _get_env(key: str, default: str | None = None) -> str:
    """Get environment variable or raise error if required and missing"""
    value = os.getenv(key, default)
    if value is None:
        print(f"Missing required environment variable: {key}", file=sys.stderr)
        sys.exit(1)
    return value


def load_config() -> Config:
    """
    Load and validate all configuration from environment variables.
    Exits with error if validation fails.
    """
    project_root = Path(__file__).parent.parent

    try:
        config = Config(
            mcp=MCPConfig(
                server_name=_get_env("MCP_SERVER_NAME", "crypto-predictor"),
                server_host=_get_env("MCP_SERVER_HOST", "localhost"),
                server_port=int(_get_env("MCP_SERVER_PORT", "4000")),
            ),
            model=ModelConfig(
                model_dir=project_root / "models" / "artifacts",
                price_only_model_name=_get_env("PRICE_ONLY_MODEL_NAME", "price_only_lgbm.pkl"),
                price_text_model_name=_get_env("PRICE_TEXT_MODEL_NAME", "price_text_lgbm.pkl"),
                transformer_model_name=_get_env("TRANSFORMER_MODEL_NAME", "transformer.pt"),
            ),
            data=DataConfig(
                price_data_source=_get_env("PRICE_DATA_SOURCE", "api"),
                raw_data_dir=project_root / "data" / "raw",
                processed_data_dir=project_root / "data" / "processed",
            ),
            app=AppConfig(
                app_env=_get_env("APP_ENV", "development"),
                log_level=_get_env("LOG_LEVEL", "debug"),
            ),
            news=NewsConfig(
                model_type=_get_env("NEWS_MODEL_TYPE", "mock"),
                max_length=int(_get_env("NEWS_MAX_LENGTH", "2000")),
            ),
            api=APIConfig(
                host=_get_env("API_HOST", "0.0.0.0"),
                port=int(_get_env("API_PORT", "5000")),
                debug=_get_env("API_DEBUG", "true").lower() == "true",
            ),
            ollama=OllamaConfig(
                host=_get_env("OLLAMA_HOST", "http://localhost:11434"),
                model=_get_env("OLLAMA_MODEL", "llama3.2"),
                timeout_ms=int(_get_env("OLLAMA_TIMEOUT_MS", "60000")),
                api_key=_get_env("OLLAMA_API_KEY", ""),
            ),
        )
        return config
    except Exception as e:
        print(f"Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)


# Global config instance - import this
env = load_config()
