import requests
from datetime import datetime

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
    session.get("https://www.nseindia.com", headers=headers)
    
    try:
        # Fetch stock data
        response = session.get(url, headers=headers, timeout=10)
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


# Example usage
if __name__ == "__main__":
    # List of stock symbols to check
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'SBIN']
    
    print("NSE Stock Prices")
    print("=" * 80)
    
    for stock in stocks:
        print(f"\nFetching data for {stock}...")
        result = get_nse_stock_price(stock)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Company: {result['company_name']}")
            print(f"Current Price: ₹{result['last_price']}")
            print(f"Change: {result['change']} ({result['percent_change']}%)")
            print(f"Open: ₹{result['open']} | High: ₹{result['high']} | Low: ₹{result['low']}")
            print(f"Previous Close: ₹{result['previous_close']}")
            print(f"Time: {result['timestamp']}")