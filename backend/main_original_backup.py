# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
import uvicorn
from pydantic import SecretStr, BaseModel
import httpx
import os
import requests
import re
from datetime import datetime
import json
import yfinance as yf
import pandas as pd
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
import pandas_ta as ta

# Define the state schema
class ChatState(TypedDict):
    messages: list  # Stores conversation history
    last_user_message: str  # Latest user input for routing
    needs_stock_data: bool  # Flag to indicate if stock data is needed
    needs_technical_analysis: bool  # Flag for technical analysis
    stock_symbol: str  # Extracted stock symbol if applicable
    analysis_period: str  # Time period for analysis (1m, 5m, 1h, 1d, etc.)


# Initialize LLM and memory checkpointer
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
GENAILAB_API_KEY = SecretStr(os.getenv("GENAILAB_API_KEY") or "")

client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-35-turbo",
    # api_key=GENAILAB_API_KEY,
    api_key="sk--17WECoVy-pTpI_dRbdSvQ",
    http_client=client
)
requests.get = lambda *args, **kwargs: requests.api.get(*args, verify=False, **kwargs)

memory = MemorySaver()


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


def get_nse_stock_price(symbol):
    """
    Fetch current stock price for NSE listed stocks
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    
    Returns:
        Dictionary with stock price information
    """
    
    # NSE API endpoint
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    
    # Headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    # Create session to handle cookies
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, verify=False)
    
    try:
        # Fetch stock data
        response = session.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information
        price_info = data.get('priceInfo', {})
        
        result = {
            'symbol': symbol,
            'company_name': data.get('info', {}).get('companyName', 'N/A'),
            'last_price': price_info.get('lastPrice', 'N/A'),
            'change': price_info.get('change', 'N/A'),
            'percent_change': price_info.get('pChange', 'N/A'),
            'open': price_info.get('open', 'N/A'),
            'high': price_info.get('intraDayHighLow', {}).get('max', 'N/A'),
            'low': price_info.get('intraDayHighLow', {}).get('min', 'N/A'),
            'previous_close': price_info.get('previousClose', 'N/A'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to fetch data: {str(e)}'}
    except KeyError as e:
        return {'error': f'Data format error: {str(e)}'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}


def get_stock_historical_data(symbol, period="3mo", interval="1d"):
    """
    Fetch historical stock data for technical analysis
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS', 'TCS.NS')
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        # Add .NS suffix for NSE stocks if not present
        if not symbol.endswith('.NS') and symbol not in ['MSFT', 'AAPL', 'GOOGL']:
            symbol = f"{symbol}.NS"
            
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return None
            
        return hist
        
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None


def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock data
    
    Args:
        df: pandas DataFrame with OHLCV data
        
    Returns:
        Dictionary with calculated indicators
    """
    if df is None or df.empty:
        return None
        
    try:
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = df['Close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = df['Close'].rolling(window=50).mean().iloc[-1]
        indicators['sma_200'] = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        
        # Exponential Moving Averages
        indicators['ema_12'] = df['Close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = df['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI (Relative Strength Index)
        if TALIB_AVAILABLE:
            rsi = talib.RSI(df['Close'].values, timeperiod=14)
            indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else None
        else:
            # Calculate RSI manually using pandas-ta
            rsi_series = ta.rsi(df['Close'], length=14)
            indicators['rsi'] = rsi_series.iloc[-1] if not rsi_series.empty else None
        
        # MACD (Moving Average Convergence Divergence)
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else None
            indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else None
            indicators['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else None
        else:
            # Calculate MACD manually
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = histogram.iloc[-1]
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma_bb = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        indicators['bb_upper'] = (sma_bb + bb_std * bb_std_dev).iloc[-1]
        indicators['bb_lower'] = (sma_bb - bb_std * bb_std_dev).iloc[-1]
        indicators['bb_middle'] = sma_bb.iloc[-1]
        
        # Volume indicators
        indicators['volume_sma_10'] = df['Volume'].rolling(window=10).mean().iloc[-1]
        indicators['current_volume'] = df['Volume'].iloc[-1]
        
        # Current price
        indicators['current_price'] = df['Close'].iloc[-1]
        indicators['previous_close'] = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return None


def generate_trading_signals(indicators, symbol):
    """
    Generate buy/sell/hold signals based on technical indicators
    
    Args:
        indicators: Dictionary of calculated technical indicators
        symbol: Stock symbol
        
    Returns:
        Dictionary with signals and analysis
    """
    if not indicators:
        return None
        
    signals = {
        'overall_signal': 'HOLD',
        'confidence': 'LOW',
        'signals': [],
        'analysis': []
    }
    
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0
    
    current_price = indicators['current_price']
    
    # RSI Analysis
    if indicators.get('rsi'):
        total_signals += 1
        rsi = indicators['rsi']
        if rsi < 30:
            signals['signals'].append("📈 RSI Oversold (Bullish)")
            signals['analysis'].append(f"RSI: {rsi:.2f} - Stock is oversold, potential buying opportunity")
            bullish_signals += 1
        elif rsi > 70:
            signals['signals'].append("📉 RSI Overbought (Bearish)")
            signals['analysis'].append(f"RSI: {rsi:.2f} - Stock is overbought, potential selling opportunity")
            bearish_signals += 1
        else:
            signals['analysis'].append(f"RSI: {rsi:.2f} - Neutral territory")
    
    # MACD Analysis
    if indicators.get('macd') and indicators.get('macd_signal'):
        total_signals += 1
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators.get('macd_histogram', 0)
        
        if macd > macd_signal and macd_hist > 0:
            signals['signals'].append("📈 MACD Bullish Crossover")
            signals['analysis'].append("MACD line above signal line - Bullish momentum")
            bullish_signals += 1
        elif macd < macd_signal and macd_hist < 0:
            signals['signals'].append("📉 MACD Bearish Crossover")
            signals['analysis'].append("MACD line below signal line - Bearish momentum")
            bearish_signals += 1
        else:
            signals['analysis'].append("MACD: Neutral - No clear crossover signal")
    
    # Moving Average Analysis
    if indicators.get('sma_20') and indicators.get('sma_50'):
        total_signals += 1
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        if current_price > sma_20 > sma_50:
            signals['signals'].append("📈 Price Above Moving Averages")
            signals['analysis'].append("Price above SMA 20 & 50 - Bullish trend")
            bullish_signals += 1
        elif current_price < sma_20 < sma_50:
            signals['signals'].append("📉 Price Below Moving Averages")
            signals['analysis'].append("Price below SMA 20 & 50 - Bearish trend")
            bearish_signals += 1
        else:
            signals['analysis'].append("Moving Averages: Mixed signals")
    
    # Bollinger Bands Analysis
    if indicators.get('bb_upper') and indicators.get('bb_lower'):
        total_signals += 1
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        if current_price >= bb_upper:
            signals['signals'].append("⚠️ Price at Upper Bollinger Band")
            signals['analysis'].append("Price touching upper band - Potential reversal zone")
            bearish_signals += 0.5
        elif current_price <= bb_lower:
            signals['signals'].append("💎 Price at Lower Bollinger Band")
            signals['analysis'].append("Price touching lower band - Potential bounce opportunity")
            bullish_signals += 0.5
    
    # Volume Analysis
    if indicators.get('current_volume') and indicators.get('volume_sma_10'):
        volume_ratio = indicators['current_volume'] / indicators['volume_sma_10']
        if volume_ratio > 1.5:
            signals['analysis'].append(f"High Volume: {volume_ratio:.1f}x average - Strong interest")
        elif volume_ratio < 0.5:
            signals['analysis'].append(f"Low Volume: {volume_ratio:.1f}x average - Weak interest")
    
    # Calculate overall signal
    if total_signals > 0:
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        if bullish_ratio > 0.6:
            signals['overall_signal'] = 'BUY'
            signals['confidence'] = 'HIGH' if bullish_ratio > 0.8 else 'MEDIUM'
        elif bearish_ratio > 0.6:
            signals['overall_signal'] = 'SELL'
            signals['confidence'] = 'HIGH' if bearish_ratio > 0.8 else 'MEDIUM'
        else:
            signals['overall_signal'] = 'HOLD'
            signals['confidence'] = 'MEDIUM'
    
    return signals


def extract_stock_symbol_from_message(message: str) -> str:
    """
    Extract stock symbol from user message using company name mapping
    
    Args:
        message: User input message
        
    Returns:
        NSE stock symbol or None if not found
    """
    message_lower = message.lower()
    
    # Check for direct stock symbol mentions (3-4 letter codes in caps)
    direct_symbols = re.findall(r'\b[A-Z]{3,4}\b', message)
    for symbol in direct_symbols:
        if symbol in COMPANY_TO_SYMBOL.values():
            return symbol
    
    # Check for company names in the mapping
    for company_name, symbol in COMPANY_TO_SYMBOL.items():
        if company_name in message_lower:
            return symbol
    
    return None


def detect_stock_query(message: str) -> bool:
    """
    Detect if user message is asking for stock price information
    
    Args:
        message: User input message
        
    Returns:
        Boolean indicating if this is a stock price query
    """
    stock_keywords = [
        "stock price", "share price", "current price", "stock quote",
        "price of", "stock value", "market price", "trading price",
        "share value", "equity price", "stock data", "price today",
        "current rate", "ltp", "last traded price"
    ]
    
    message_lower = message.lower()
    
    # Check for stock-related keywords
    has_stock_keyword = any(keyword in message_lower for keyword in stock_keywords)
    
    # Check if a company/symbol is mentioned
    has_company = extract_stock_symbol_from_message(message) is not None
    
    return has_stock_keyword and has_company


def detect_technical_analysis_query(message: str) -> bool:
    """
    Detect if user message is asking for technical analysis
    
    Args:
        message: User input message
        
    Returns:
        Boolean indicating if this is a technical analysis query
    """
    technical_keywords = [
        "technical analysis", "chart analysis", "ta", "rsi", "macd",
        "moving average", "bollinger bands", "support", "resistance",
        "buy signal", "sell signal", "trading signal", "trend analysis",
        "momentum", "volume analysis", "chart pattern", "indicators"
    ]
    
    message_lower = message.lower()
    
    # Check for technical analysis keywords
    has_technical_keyword = any(keyword in message_lower for keyword in technical_keywords)
    
    # Check if a company/symbol is mentioned
    has_company = extract_stock_symbol_from_message(message) is not None
    
    return has_technical_keyword and has_company


def extract_analysis_period(message: str) -> str:
    """
    Extract time period for analysis from user message
    
    Args:
        message: User input message
        
    Returns:
        Time period string (default: "3mo")
    """
    period_mapping = {
        "1 day": "1d", "1day": "1d", "today": "1d",
        "1 week": "5d", "1week": "5d", "week": "5d",
        "1 month": "1mo", "1month": "1mo", "month": "1mo",
        "3 months": "3mo", "3months": "3mo", "quarter": "3mo",
        "6 months": "6mo", "6months": "6mo", "half year": "6mo",
        "1 year": "1y", "1year": "1y", "year": "1y", "yearly": "1y",
        "2 years": "2y", "2years": "2y",
        "5 years": "5y", "5years": "5y"
    }
    
    message_lower = message.lower()
    
    for phrase, period in period_mapping.items():
        if phrase in message_lower:
            return period
    
    return "3mo"  # Default period


def router_node(state: ChatState) -> ChatState:
    """
    Route user queries to appropriate handlers based on content analysis
    """
    messages = state["messages"]
    if not messages:
        return state
        
    # Get the last user message
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state
    
    user_message = last_message.content
    state["last_user_message"] = user_message
    
    # Check for technical analysis query first (higher priority)
    if detect_technical_analysis_query(user_message):
        state["needs_technical_analysis"] = True
        state["needs_stock_data"] = False
        state["stock_symbol"] = extract_stock_symbol_from_message(user_message)
        state["analysis_period"] = extract_analysis_period(user_message)
    # Check for stock price query
    elif detect_stock_query(user_message):
        state["needs_stock_data"] = True
        state["needs_technical_analysis"] = False
        state["stock_symbol"] = extract_stock_symbol_from_message(user_message)
        state["analysis_period"] = ""
    else:
        state["needs_stock_data"] = False
        state["needs_technical_analysis"] = False
        state["stock_symbol"] = ""
        state["analysis_period"] = ""
    
    return state


def stock_agent_node(state: ChatState) -> ChatState:
    """
    Fetch stock price data and format response
    """
    symbol = state.get("stock_symbol", "")
    user_message = state.get("last_user_message", "")
    
    if not symbol:
        # Try to extract symbol again
        symbol = extract_stock_symbol_from_message(user_message)
    
    if not symbol:
        # Add error message if no symbol found
        error_message = AIMessage(
            content="I couldn't identify which stock you're asking about. Please specify a valid NSE listed company name or stock symbol (e.g., TCS, RELIANCE, INFY)."
        )
        state["messages"].append(error_message)
        return state
    
    # Fetch stock data
    print(f"[Stock Agent] Fetching data for symbol: {symbol}")
    stock_data = get_nse_stock_price(symbol)
    
    if 'error' in stock_data:
        error_response = AIMessage(
            content=f"Sorry, I couldn't fetch the stock price for {symbol}. Error: {stock_data['error']}"
        )
        state["messages"].append(error_response)
        return state
    
    # Format the response
    response_text = f"""📈 **Stock Price Information for {stock_data['company_name']} ({stock_data['symbol']})**

💰 **Current Price**: ₹{stock_data['last_price']}
📊 **Change**: ₹{stock_data['change']} ({stock_data['percent_change']}%)
🔄 **Today's Range**: ₹{stock_data['low']} - ₹{stock_data['high']}
🌅 **Opening Price**: ₹{stock_data['open']}
🕰️ **Previous Close**: ₹{stock_data['previous_close']}
⏰ **Last Updated**: {stock_data['timestamp']}

{"📈 Stock is UP today!" if float(stock_data.get('change', 0)) > 0 else "📉 Stock is DOWN today!" if float(stock_data.get('change', 0)) < 0 else "➡️ Stock is FLAT today!"}"""

    stock_response = AIMessage(content=response_text)
    state["messages"].append(stock_response)
    
    return state


def technical_analysis_agent_node(state: ChatState) -> ChatState:
    """
    Advanced stock technical analysis and predictions
    
    Features: RSI, MACD, moving averages, buy/sell signals
    """
    symbol = state.get("stock_symbol", "")
    user_message = state.get("last_user_message", "")
    analysis_period = state.get("analysis_period", "3mo")
    
    if not symbol:
        # Try to extract symbol again
        symbol = extract_stock_symbol_from_message(user_message)
    
    if not symbol:
        # Add error message if no symbol found
        error_message = AIMessage(
            content="I couldn't identify which stock you want to analyze. Please specify a valid NSE listed company name or stock symbol (e.g., TCS, RELIANCE, INFY)."
        )
        state["messages"].append(error_message)
        return state
    
    print(f"[Technical Analysis Agent] Analyzing {symbol} for period {analysis_period}")
    
    # Fetch historical data
    historical_data = get_stock_historical_data(symbol, period=analysis_period)
    
    if historical_data is None or historical_data.empty:
        error_response = AIMessage(
            content=f"Sorry, I couldn't fetch historical data for {symbol} to perform technical analysis. Please try again or check if the symbol is correct."
        )
        state["messages"].append(error_response)
        return state
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(historical_data)
    
    if not indicators:
        error_response = AIMessage(
            content=f"Sorry, I couldn't calculate technical indicators for {symbol}. Please try again later."
        )
        state["messages"].append(error_response)
        return state
    
    # Generate trading signals
    signals = generate_trading_signals(indicators, symbol)
    
    # Format the response
    current_price = indicators['current_price']
    previous_close = indicators['previous_close']
    price_change = current_price - previous_close
    price_change_pct = (price_change / previous_close) * 100
    
    # Create comprehensive technical analysis report
    response_text = f"""📊 **Technical Analysis Report for {symbol}**

💰 **Current Price**: ₹{current_price:.2f} ({'+' if price_change >= 0 else ''}{price_change:.2f}, {price_change_pct:.2f}%)
📈 **Overall Signal**: {signals['overall_signal']} ({signals['confidence']} confidence)

🔍 **Technical Indicators**:
• RSI (14): {indicators.get('rsi', 'N/A'):.2f if indicators.get('rsi') else 'N/A'}
• MACD: {indicators.get('macd', 'N/A'):.4f if indicators.get('macd') else 'N/A'}
• Signal Line: {indicators.get('macd_signal', 'N/A'):.4f if indicators.get('macd_signal') else 'N/A'}

📊 **Moving Averages**:
• SMA 20: ₹{indicators.get('sma_20', 'N/A'):.2f if indicators.get('sma_20') else 'N/A'}
• SMA 50: ₹{indicators.get('sma_50', 'N/A'):.2f if indicators.get('sma_50') else 'N/A'}
• EMA 12: ₹{indicators.get('ema_12', 'N/A'):.2f if indicators.get('ema_12') else 'N/A'}

🎯 **Bollinger Bands**:
• Upper: ₹{indicators.get('bb_upper', 'N/A'):.2f if indicators.get('bb_upper') else 'N/A'}
• Middle: ₹{indicators.get('bb_middle', 'N/A'):.2f if indicators.get('bb_middle') else 'N/A'}
• Lower: ₹{indicators.get('bb_lower', 'N/A'):.2f if indicators.get('bb_lower') else 'N/A'}

📈 **Key Signals**:
{chr(10).join(f"• {signal}" for signal in signals['signals']) if signals['signals'] else "• No clear signals at this time"}

📝 **Detailed Analysis**:
{chr(10).join(f"• {analysis}" for analysis in signals['analysis'])}

⚠️ **Disclaimer**: This is for educational purposes only. Always do your own research and consider consulting a financial advisor before making investment decisions.
"""

    technical_response = AIMessage(content=response_text)
    state["messages"].append(technical_response)
    
    return state


# Node function that processes regular chat messages
def chatbot_node(state: ChatState) -> ChatState:
    """Process user message and generate response for general queries."""
    # Get the conversation history for context
    messages = state["messages"]
    
    # Call the LLM with full conversation history
    response = llm.invoke(messages)
    
    # Append AI response to messages
    messages.append(AIMessage(content=response.content))
    
    return {"messages": messages}


# Enhanced conditional routing function
def route_to_agent(state: ChatState) -> str:
    """Determine which node to route to based on user query type"""
    if state.get("needs_technical_analysis", False):
        return "technical_analysis_agent"
    elif state.get("needs_stock_data", False):
        return "stock_agent"
    else:
        return "chatbot"


# Build the graph with intelligent routing
graph_builder = StateGraph(ChatState)

# Add nodes
graph_builder.add_node("router", router_node)
graph_builder.add_node("chatbot", chatbot_node) 
graph_builder.add_node("stock_agent", stock_agent_node)
graph_builder.add_node("technical_analysis_agent", technical_analysis_agent_node)

# Define edges with conditional routing
graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges(
    "router",
    route_to_agent,
    {
        "chatbot": "chatbot",
        "stock_agent": "stock_agent",
        "technical_analysis_agent": "technical_analysis_agent"
    }
)
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge("stock_agent", END)
graph_builder.add_edge("technical_analysis_agent", END)

# Compile with memory persistence
graph = graph_builder.compile(checkpointer=memory)


# Pydantic models for API
class MessageInput(BaseModel):
    text: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# FastAPI app
app = FastAPI(title="LangGraph Chatbot", version="1.0.0")

# Add CORS middleware to allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(message: MessageInput, thread_id: str = "default"):
    """
    Send a message to the chatbot.
    
    Args:
        message: MessageInput with 'text' field
        thread_id: Unique identifier for conversation thread (defaults to "default")
    
    Returns:
        ChatResponse with the AI response and thread_id
    """
    try:
        # Prepare input with thread configuration and initialize state
        user_input = {
            "messages": [HumanMessage(content=message.text)],
            "last_user_message": message.text,
            "needs_stock_data": False,
            "needs_technical_analysis": False,
            "stock_symbol": "",
            "analysis_period": ""
        }
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the graph with persistence
        # If thread exists, it loads previous state; otherwise creates new one
        output = graph.invoke(user_input, config)
        
        # Extract the last AI response
        last_message = output["messages"][-1]
        
        return ChatResponse(
            response=last_message.content,
            thread_id=thread_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Retrieve conversation history for a thread.
    
    Args:
        thread_id: Unique identifier for conversation thread
    
    Returns:
        Full message history for the thread
    """
    try:
        # Get the current state from memory
        state_snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        
        if not state_snapshot.values:
            return {"messages": [], "thread_id": thread_id}
        
        # Format messages for response
        messages = []
        for msg in state_snapshot.values.get("messages", []):
            messages.append({
                "role": "user" if hasattr(msg, '__class__') and "Human" in msg.__class__.__name__ else "assistant",
                "content": msg.content
            })
        
        return {"messages": messages, "thread_id": thread_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/reset/{thread_id}")
async def reset_conversation(thread_id: str):
    """
    Clear conversation history for a thread.
    
    Args:
        thread_id: Unique identifier for conversation thread
    
    Returns:
        Success message
    """
    try:
        # Create empty state for the thread
        graph.invoke(
            {"messages": []},
            {"configurable": {"thread_id": thread_id}}
        )
        return {"message": f"Thread {thread_id} reset successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{symbol}")
async def get_stock_price_endpoint(symbol: str):
    """
    Get current stock price for a specific symbol
    
    Args:
        symbol: NSE stock symbol (e.g., TCS, RELIANCE, INFY)
    
    Returns:
        Stock price information
    """
    try:
        # Convert to uppercase for consistency
        symbol = symbol.upper()
        
        # Fetch stock data
        stock_data = get_nse_stock_price(symbol)
        
        if 'error' in stock_data:
            raise HTTPException(status_code=404, detail=f"Stock data not found for {symbol}: {stock_data['error']}")
        
        return stock_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies")
async def get_supported_companies():
    """
    Get list of supported companies and their stock symbols
    
    Returns:
        Dictionary of company names mapped to stock symbols
    """
    return {
        "supported_companies": COMPANY_TO_SYMBOL,
        "total_companies": len(COMPANY_TO_SYMBOL)
    }


@app.get("/technical-analysis/{symbol}")
async def get_technical_analysis_endpoint(symbol: str, period: str = "3mo"):
    """
    Get technical analysis for a specific symbol
    
    Args:
        symbol: NSE stock symbol (e.g., TCS, RELIANCE, INFY)
        period: Analysis period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    Returns:
        Technical analysis with indicators and signals
    """
    try:
        # Convert to uppercase for consistency
        symbol = symbol.upper()
        
        # Fetch historical data
        historical_data = get_stock_historical_data(symbol, period=period)
        
        if historical_data is None or historical_data.empty:
            raise HTTPException(status_code=404, detail=f"Historical data not found for {symbol}")
        
        # Calculate indicators
        indicators = calculate_technical_indicators(historical_data)
        if not indicators:
            raise HTTPException(status_code=500, detail="Failed to calculate technical indicators")
        
        # Generate signals
        signals = generate_trading_signals(indicators, symbol)
        
        return {
            "symbol": symbol,
            "period": period,
            "current_price": indicators['current_price'],
            "indicators": indicators,
            "signals": signals,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the service is running
    
    Returns:
        Service status information
    """
    return {
        "status": "healthy",
        "service": "LangGraph ChatBot with Stock & Technical Analysis Agents",
        "version": "2.0.0",
        "features": [
            "AI Chat with conversation memory",
            "NSE Stock price lookup",
            "Technical Analysis (RSI, MACD, Moving Averages)",
            "Trading signals and recommendations", 
            "Intelligent query routing",
            "Multi-threaded conversations"
        ],
        "supported_indicators": ["RSI", "MACD", "SMA", "EMA", "Bollinger Bands", "Volume Analysis"],
        "supported_stock_exchanges": ["NSE (National Stock Exchange of India)"]
    }


@app.get("/", response_class=HTMLResponse)
async def serve_test_page():
    """Serve the test HTML page"""
    try:
        with open("test_chat.html", "r", encoding="utf-8") as file:
            html_content = file.read()
            # Update the API base URL in the HTML
            html_content = html_content.replace("const API_BASE = 'http://127.0.0.1:8000';", "const API_BASE = 'http://127.0.0.1:8001';")
            return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)