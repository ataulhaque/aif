# models.py
"""
Data models and state definitions for the ChatBot application
"""

from pydantic import BaseModel
from typing_extensions import TypedDict


# Define the state schema for LangGraph
class ChatState(TypedDict):
    messages: list  # Stores conversation history
    last_user_message: str  # Latest user input for routing
    needs_stock_data: bool  # Flag to indicate if stock data is needed
    needs_technical_analysis: bool  # Flag for technical analysis
    stock_symbol: str  # Extracted stock symbol if applicable
    analysis_period: str  # Time period for analysis (1m, 5m, 1h, 1d, etc.)


# Pydantic models for API
class MessageInput(BaseModel):
    text: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class StockPriceResponse(BaseModel):
    symbol: str
    company_name: str
    last_price: str
    change: str
    percent_change: str
    open: str
    high: str
    low: str
    previous_close: str
    timestamp: str


class TechnicalAnalysisResponse(BaseModel):
    symbol: str
    period: str
    current_price: float
    indicators: dict
    signals: dict
    timestamp: str
