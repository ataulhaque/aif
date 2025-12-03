# config.py
"""
Configuration settings and constants for the ChatBot application
"""

import os
from pydantic import SecretStr

# API Configuration
GENAILAB_API_KEY = SecretStr(os.getenv("GENAILAB_API_KEY") or "sk--17WECoVy-pTpI_dRbdSvQ")
GENAILAB_BASE_URL = "https://genailab.tcs.in"
GENAILAB_MODEL = "azure/genailab-maas-gpt-35-turbo"

# NSE Stock Price Configuration
NSE_BASE_URL = "https://www.nseindia.com"
NSE_QUOTE_URL = f"{NSE_BASE_URL}/api/quote-equity"

# Company name to NSE symbol mapping
COMPANY_TO_SYMBOL = {
    "tata consultancy services": "TCS",
    "tcs": "TCS",
    "infosys": "INFY",
    "reliance": "RELIANCE",
    "reliance industries": "RELIANCE",
    "hdfc bank": "HDFCBANK",
    "hdfc": "HDFCBANK", 
    "state bank of india": "SBIN",
    "sbi": "SBIN",
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "itc": "ITC",
    "wipro": "WIPRO",
    "hcl technologies": "HCLTECH",
    "hcl": "HCLTECH",
    "tech mahindra": "TECHM",
    "bajaj finance": "BAJFINANCE",
    "maruti suzuki": "MARUTI",
    "maruti": "MARUTI",
    "asian paints": "ASIANPAINT",
    "nestle": "NESTLEIND",
    "microsoft": "MSFT",  # For international stocks (if supported)
    "apple": "AAPL"
}

# Technical Analysis Configuration
DEFAULT_ANALYSIS_PERIOD = "3mo"
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BOLLINGER_BANDS_PERIOD = 20
BOLLINGER_BANDS_STD = 2

# Headers for NSE requests
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Stock keywords for detection
STOCK_KEYWORDS = [
    "stock price", "share price", "current price", "stock quote",
    "price of", "stock value", "market price", "trading price",
    "share value", "equity price", "stock data", "price today",
    "current rate", "ltp", "last traded price"
]

# Technical analysis keywords for detection
TECHNICAL_KEYWORDS = [
    "technical analysis", "chart analysis", "ta", "rsi", "macd",
    "moving average", "bollinger bands", "support", "resistance",
    "buy signal", "sell signal", "trading signal", "trend analysis",
    "momentum", "volume analysis", "chart pattern", "indicators"
]

# Period mapping for analysis
PERIOD_MAPPING = {
    "1 day": "1d", "1day": "1d", "today": "1d",
    "1 week": "5d", "1week": "5d", "week": "5d",
    "1 month": "1mo", "1month": "1mo", "month": "1mo",
    "3 months": "3mo", "3months": "3mo", "quarter": "3mo",
    "6 months": "6mo", "6months": "6mo", "half year": "6mo",
    "1 year": "1y", "1year": "1y", "year": "1y", "yearly": "1y",
    "2 years": "2y", "2years": "2y",
    "5 years": "5y", "5years": "5y"
}
