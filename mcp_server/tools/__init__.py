"""
MCP Tools module.
Contains tool definitions for the fastmcp server.
"""

from pydantic import BaseModel, field_validator
from typing import Literal


class PredictPriceInput(BaseModel):
    """Input schema for predict_price_price_only tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    as_of: str  # YYYY-MM-DD format or "now"

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, v: str) -> str:
        if v == "now":
            return v
        # Validate date format
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("as_of must be 'now' or YYYY-MM-DD format")
        return v


class PredictPriceOutput(BaseModel):
    """Output schema for predict_price_price_only tool"""
    symbol: str
    as_of: str
    prediction: Literal["UP", "DOWN"]
    prob_up: float
    model_variant: str
    features_used: list[str]


# ===== Phase 4: Price + Text Tool =====

class PredictPriceWithTextInput(BaseModel):
    """Input schema for predict_price_with_text tool"""
    symbol: Literal["BTC", "ETH", "SOL"]
    as_of: str  # YYYY-MM-DD format or "now"
    news_text: str  # News text for sentiment analysis

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, v: str) -> str:
        if v == "now":
            return v
        # Validate date format
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("as_of must be 'now' or YYYY-MM-DD format")
        return v

    @field_validator("news_text")
    @classmethod
    def validate_news_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("news_text cannot be empty")
        return v.strip()


class PredictPriceWithTextOutput(BaseModel):
    """Output schema for predict_price_with_text tool"""
    symbol: str
    as_of: str
    prediction: Literal["UP", "DOWN"]
    prob_up: float
    model_variant: str
    features_used: list[str]
    sentiment_score: float  # [-1.0, 1.0]
