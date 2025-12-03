import requests
import ssl
import urllib3
import urllib.parse
import json
from datetime import datetime

NEWS_API_KEY = "ad42918c892e452f98c8d115eb9a2ae1"

# --- SSL WORKAROUND (Required by specific setup) ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_news(company_name):
    """Fetch news for a specific company"""
    # URL encode the query string
    encoded_query = urllib.parse.quote(company_name)
    
    print(f"[NewsAgent Tool] Fetching news for keywords: {company_name}")
    
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={encoded_query}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"pageSize=10&"
        f"apiKey={NEWS_API_KEY}"
    )
    
    try:
        # Use requests with SSL verification disabled
        response = requests.get(url, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if not data or data.get("status") != "ok":
            print("No news found or API error.")
            return None
        
        articles = data.get("articles", [])
        formatted_news = []
        
        for article in articles:
            formatted_news.append({
                "title": article.get('title'),
                "url": article.get('url'),
                "description": article.get('description', ''),
                "published_at": article.get('publishedAt'),
                "source": article.get('source', {}).get('name', 'Unknown')
            })
        
        return formatted_news
        
    except Exception as e:
        print(f"[NewsAgent Error] NewsAPI request failed: {e}")
        return None

def get_stock_info_alternative(symbol):
    """Alternative stock info using Alpha Vantage API (free tier available)"""
    # Note: You would need an Alpha Vantage API key for this
    # For demonstration, we'll return mock data
    print(f"Fetching stock info for {symbol}...")
    
    # Mock stock data structure
    stock_info = {
        "symbol": symbol,
        "company_name": f"{symbol} Corporation",
        "current_price": "250.75",
        "change": "+2.50",
        "change_percent": "+1.00%",
        "volume": "1,234,567",
        "market_cap": "1.5T",
        "pe_ratio": "25.4",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return stock_info

if __name__ == "__main__":
    # Test news fetching
    company = "Tata Consultancy Services Limited"
    news = fetch_news(company)
    
    if news:
        print(f"\n=== News for {company} ===")
        for i, article in enumerate(news[:3], 1):  # Show top 3 articles
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Published: {article['published_at']}")
            print(f"   URL: {article['url']}")
            print(f"   Description: {article['description'][:100]}...")
    
    # Test stock info
    stock_data = get_stock_info_alternative("MSFT")
    print(f"\n=== Stock Info for {stock_data['symbol']} ===")
    for key, value in stock_data.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
