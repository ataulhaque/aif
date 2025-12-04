# technical_analysis_agent.py
"""
Technical analysis agent for advanced stock analysis and trading signals
"""

from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from models import ChatState
from utils import extract_stock_symbol_from_message
from technical_analysis import get_stock_historical_data, calculate_technical_indicators, generate_trading_signals


def technical_analysis_agent_node(state: ChatState) -> ChatState:
    """
    Advanced stock technical analysis and predictions
    
    Features: RSI, MACD, moving averages, buy/sell signals
    
    Args:
        state: Current chat state
        
    Returns:
        Updated chat state with technical analysis response
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
    indicators = calculate_technical_indicators(historical_data, symbol)
    
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
    
    # Safely format indicators with proper null checks
    def safe_format(value, decimals=2):
        return f"{value:.{decimals}f}" if value is not None else "N/A"
    
    def safe_format_currency(value, decimals=2):
        return f"₹{value:.{decimals}f}" if value is not None else "N/A"
    
    # Create comprehensive technical analysis report
    response_text = f"""📊 **Technical Analysis Report for {symbol}**

💰 **Current Price**: ₹{current_price:.2f} ({'+' if price_change >= 0 else ''}{price_change:.2f}, {price_change_pct:.2f}%)
📈 **Overall Signal**: {signals['overall_signal']} ({signals['confidence']} confidence)

🔍 **Technical Indicators**:
• RSI (14): {safe_format(indicators.get('rsi'))}
• MACD: {safe_format(indicators.get('macd'), 4)}
• Signal Line: {safe_format(indicators.get('macd_signal'), 4)}

📊 **Moving Averages**:
• SMA 20: {safe_format_currency(indicators.get('sma_20'))}
• SMA 50: {safe_format_currency(indicators.get('sma_50'))}
• EMA 12: {safe_format_currency(indicators.get('ema_12'))}

🎯 **Bollinger Bands**:
• Upper: {safe_format_currency(indicators.get('bb_upper'))}
• Middle: {safe_format_currency(indicators.get('bb_middle'))}
• Lower: {safe_format_currency(indicators.get('bb_lower'))}

📈 **Key Signals**:
{chr(10).join(f"• {signal}" for signal in signals['signals']) if signals['signals'] else "• No clear signals at this time"}

📝 **Detailed Analysis**:
{chr(10).join(f"• {analysis}" for analysis in signals['analysis'])}

⚠️ **Disclaimer**: This is for educational purposes only. Always do your own research and consider consulting a financial advisor before making investment decisions.
"""

    technical_response = AIMessage(content=response_text)
    state["messages"].append(technical_response)
    
    return state
