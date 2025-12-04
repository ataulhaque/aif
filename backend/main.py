# main.py
"""
Main FastAPI application orchestrator for the modular ChatBot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime
from langchain_core.messages import HumanMessage

# Import modular components
from models import MessageInput, ChatResponse
from config import COMPANY_TO_SYMBOL
from utils import setup_ssl_bypass, get_nse_stock_price
from technical_analysis import get_stock_historical_data, calculate_technical_indicators, generate_trading_signals
from graph_builder import build_chat_graph

# Setup SSL bypass for requests
setup_ssl_bypass()

# Build the conversation graph
graph = build_chat_graph()

# FastAPI app
app = FastAPI(title="LangGraph Chatbot", version="2.0.0")

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
        indicators = calculate_technical_indicators(historical_data, symbol)
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
        "architecture": "modular",
        "features": [
            "AI Chat with conversation memory",
            "NSE Stock price lookup",
            "Technical Analysis (RSI, MACD, Moving Averages)",
            "Trading signals and recommendations", 
            "Intelligent query routing",
            "Multi-threaded conversations"
        ],        "supported_indicators": ["RSI", "MACD", "SMA", "EMA", "Bollinger Bands", "Volume Analysis"],
        "supported_stock_exchanges": ["Alpha Vantage (US Markets)", "NSE (Demo Data)"],
        "data_sources": ["Alpha Vantage API", "Demo Data for NSE stocks"],
        "modules": [
            "config.py - Configuration and constants",
            "models.py - Data models and schemas",
            "utils.py - Utility functions",
            "stock_agent.py - Stock price agent",
            "technical_analysis.py - Technical analysis calculations",
            "technical_analysis_agent.py - Technical analysis agent",
            "router.py - Query routing and chatbot logic",
            "graph_builder.py - LangGraph workflow builder",
            "main.py - FastAPI orchestrator"
        ]
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
