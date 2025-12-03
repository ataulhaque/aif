import requests
import ssl
import urllib3
import urllib.parse
import yfinance as yf
import os
from get_stock_price import get_nse_stock_price

NEWS_API_KEY = "ad42918c892e452f98c8d115eb9a2ae1"

# --- SSL WORKAROUND (Required by specific setup) ---
# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variable to disable SSL verification for curl-cffi (used by yfinance)
os.environ['CURL_CA_BUNDLE'] = ''

# Override requests to disable SSL verification
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = patched_get

keyword_string = "Tata Consultancy Services Limited"

# Map company names to NSE stock symbols
COMPANY_TO_SYMBOL = {
    "Tata Consultancy Services Limited": "TCS",
    "Tata Consultancy Services": "TCS",
    "TCS": "TCS",
    "Infosys Limited": "INFY",
    "Infosys": "INFY",
    "Reliance Industries Limited": "RELIANCE",
    "Reliance": "RELIANCE",
    "HDFC Bank Limited": "HDFCBANK",
    "HDFC Bank": "HDFCBANK",
    "State Bank of India": "SBIN",
    "SBI": "SBIN"
}

def get_stock_symbol(company_name):
    """Convert company name to NSE stock symbol"""
    return COMPANY_TO_SYMBOL.get(company_name, company_name.upper())

def fetch_stock_ltp(company_name):
    """
    Fetch Last Traded Price (LTP) for the given company
    
    Args:
        company_name: Name of the company
        
    Returns:
        Dictionary with stock price information or error message
    """
    print(f"[StockAgent Tool] Fetching LTP for: {company_name}")
    
    # Get stock symbol
    stock_symbol = get_stock_symbol(company_name)
    print(f"[StockAgent Tool] Using symbol: {stock_symbol}")
    
    try:
        # Fetch stock price using the get_stock_price module
        stock_data = get_nse_stock_price(stock_symbol)
        
        if 'error' in stock_data:
            return {
                'success': False,
                'error': f"Failed to fetch stock data: {stock_data['error']}",
                'symbol': stock_symbol,
                'company': company_name
            }
        
        return {
            'success': True,
            'company': stock_data.get('company_name', company_name),
            'symbol': stock_symbol,
            'ltp': stock_data.get('last_price', 'N/A'),
            'change': stock_data.get('change', 'N/A'),
            'percent_change': stock_data.get('percent_change', 'N/A'),
            'open': stock_data.get('open', 'N/A'),
            'high': stock_data.get('high', 'N/A'),
            'low': stock_data.get('low', 'N/A'),
            'previous_close': stock_data.get('previous_close', 'N/A'),
            'timestamp': stock_data.get('timestamp', 'N/A')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Exception occurred: {str(e)}",
            'symbol': stock_symbol,
            'company': company_name
        }

# URL encode the query string
encoded_query = urllib.parse.quote(keyword_string)

print(f"[NewsAgent Tool] Fetching news for keywords: {keyword_string}")

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={encoded_query}&"
    f"sortBy=publishedAt&"
    f"language=en&"
    f"pageSize=10&"
    f"apiKey={NEWS_API_KEY}"
)

try:
    res = requests.get(url)
    res.raise_for_status()   # Raise error for bad status codes
    data = res.json()
except Exception as e:
    print(f"[NewsAgent Error] NewsAPI request failed: {e}")
    data = None

if not data or data.get("status") != "ok":
    print("No news found or API error.")
else:
    articles = data.get("articles", [])
    formatted = []
    for a in articles:
        formatted.append(
            f"TITLE: {a.get('title')}\n"
            f"URL: {a.get('url')}\n"
            f"CONTENT: {a.get('content') or a.get('description', '')}\n"
            "-------------------------\n"
        )   
    print("\n".join(formatted))

print("\n" + "="*80)
print("STOCK PRICE INFORMATION")
print("="*80)

# Fetch stock price for the company
stock_info = fetch_stock_ltp(keyword_string)

if stock_info['success']:
    print(f"📈 STOCK PRICE DATA FOR {stock_info['company'].upper()}")
    print(f"Symbol: {stock_info['symbol']}")
    print(f"Last Traded Price (LTP): ₹{stock_info['ltp']}")
    print(f"Change: ₹{stock_info['change']} ({stock_info['percent_change']}%)")
    print(f"Today's Range: ₹{stock_info['low']} - ₹{stock_info['high']}")
    print(f"Opening Price: ₹{stock_info['open']}")
    print(f"Previous Close: ₹{stock_info['previous_close']}")
    print(f"Data fetched at: {stock_info['timestamp']}")
    
    # Determine if stock is up or down
    try:
        change_value = float(stock_info['change'])
        if change_value > 0:
            print("📈 Stock is UP today!")
        elif change_value < 0:
            print("📉 Stock is DOWN today!")
        else:
            print("➡️ Stock is FLAT today!")
    except (ValueError, TypeError):
        print("ℹ️ Change information not available")
        
else:
    print(f"❌ ERROR fetching stock data for {stock_info['company']}")
    print(f"Symbol attempted: {stock_info['symbol']}")
    print(f"Error details: {stock_info['error']}")
    print("💡 Tip: Make sure the company name maps to a valid NSE stock symbol")

