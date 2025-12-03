# stock_agent.py
"""
Stock price agent for fetching real-time NSE stock prices
"""

from langchain_core.messages import AIMessage, HumanMessage
from models import ChatState
from utils import extract_stock_symbol_from_message, get_nse_stock_price


def stock_agent_node(state: ChatState) -> ChatState:
    """
    Fetch stock price data and format response
    
    Args:
        state: Current chat state
        
    Returns:
        Updated chat state with stock price response
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
