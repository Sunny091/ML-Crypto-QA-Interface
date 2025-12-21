"""
Configuration module - single source of truth for all environment variables.
All env reads happen ONLY in this file.
"""

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

# Load .env from project root (two levels up from config/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
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
    price_text_model_name: str  # Phase 4

    @property
    def price_only_model_path(self) -> Path:
        return self.model_dir / self.price_only_model_name

    @property
    def price_text_model_path(self) -> Path:
        return self.model_dir / self.price_text_model_name


class NewsConfig(BaseModel):
    """Phase 4: News analysis configuration"""
    model_type: Literal["finbert", "mock"]
    max_length: int


class DataConfig(BaseModel):
    """Data source configuration"""
    price_data_source: Literal["mock", "api", "file"]


class AppConfig(BaseModel):
    """Application configuration"""
    app_env: Literal["development", "production", "test"]
    log_level: Literal["debug", "info", "warn", "error"]

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


class Config(BaseModel):
    """Root configuration container"""
    mcp: MCPConfig
    model: ModelConfig
    data: DataConfig
    app: AppConfig
    news: NewsConfig  # Phase 4


def _get_env(key: str, default: str | None = None) -> str:
    """Get environment variable or raise error if required and missing"""
    value = os.getenv(key, default)
    if value is None:
        print(f"❌ Missing required environment variable: {key}", file=sys.stderr)
        sys.exit(1)
    return value


def load_config() -> Config:
    """
    Load and validate all configuration from environment variables.
    Exits with error if validation fails.
    """
    try:
        config = Config(
            mcp=MCPConfig(
                server_name=_get_env("MCP_SERVER_NAME", "crypto-predictor"),
                server_host=_get_env("MCP_SERVER_HOST", "localhost"),
                server_port=int(_get_env("MCP_SERVER_PORT", "4000")),
            ),
            model=ModelConfig(
                model_dir=Path(_get_env("MODEL_DIR", "./models")),
                price_only_model_name=_get_env("PRICE_ONLY_MODEL_NAME", "price_only_lgbm.pkl"),
                price_text_model_name=_get_env("PRICE_TEXT_MODEL_NAME", "price_text_lgbm.pkl"),
            ),
            data=DataConfig(
                price_data_source=_get_env("PRICE_DATA_SOURCE", "mock"),  # type: ignore
            ),
            app=AppConfig(
                app_env=_get_env("APP_ENV", "development"),  # type: ignore
                log_level=_get_env("LOG_LEVEL", "debug"),  # type: ignore
            ),
            news=NewsConfig(
                model_type=_get_env("NEWS_MODEL_TYPE", "finbert"),  # type: ignore
                max_length=int(_get_env("NEWS_MAX_LENGTH", "2000")),
            ),
        )
        return config
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)


# Global config instance - import this
env = load_config()
