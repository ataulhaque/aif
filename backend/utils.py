# utils.py
"""
Utility functions for the ChatBot application
"""

import re
import requests
from datetime import datetime
from config import COMPANY_TO_SYMBOL, NSE_HEADERS, NSE_QUOTE_URL, STOCK_KEYWORDS, TECHNICAL_KEYWORDS, PERIOD_MAPPING


def setup_ssl_bypass():
    """Setup SSL bypass for requests"""
    import ssl
    import urllib3
    
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Override requests to disable SSL verification
    original_get = requests.get
    def patched_get(*args, **kwargs):
        kwargs['verify'] = False
        return original_get(*args, **kwargs)
    requests.get = patched_get


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
    message_lower = message.lower()
    
    # Check for stock-related keywords
    has_stock_keyword = any(keyword in message_lower for keyword in STOCK_KEYWORDS)
    
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
    message_lower = message.lower()
    
    # Check for technical analysis keywords
    has_technical_keyword = any(keyword in message_lower for keyword in TECHNICAL_KEYWORDS)
    
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
    message_lower = message.lower()
    
    for phrase, period in PERIOD_MAPPING.items():
        if phrase in message_lower:
            return period
    
    return "3mo"  # Default period


def get_nse_stock_price(symbol: str) -> dict:
    """
    Fetch current stock price for NSE listed stocks
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    
    Returns:
        Dictionary with stock price information
    """
    # NSE API endpoint
    url = f"{NSE_QUOTE_URL}?symbol={symbol}"
    
    # Create session to handle cookies
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=NSE_HEADERS, verify=False)
    
    try:
        # Fetch stock data
        response = session.get(url, headers=NSE_HEADERS, timeout=10, verify=False)
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
