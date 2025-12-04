# Alpha Vantage Integration Guide

## Overview

The ChatBot has been updated to use Alpha Vantage instead of Yahoo Finance for technical analysis. This provides more reliable and comprehensive financial data with professional-grade technical indicators.

## Features

### ✅ Alpha Vantage Integration
- **Real-time Data**: Professional-grade stock data from Alpha Vantage
- **Technical Indicators**: RSI, MACD, and other indicators calculated by Alpha Vantage's servers
- **US Stock Market**: Full coverage of US stocks (NASDAQ, NYSE, AMEX)
- **API Rate Limits**: Respectful API usage with built-in delays

### 🎯 Supported Markets
- **US Markets**: Full Alpha Vantage coverage (MSFT, AAPL, GOOGL, etc.)
- **NSE Markets**: Demo data generation (TCS, RELIANCE, INFY, etc.)

## Setup Instructions

### 1. Get Alpha Vantage API Key
1. Visit [Alpha Vantage](https://www.alphavantage.co/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Free tier includes 25 requests per day

### 2. Configure API Key
Add your Alpha Vantage API key to your environment:

**Windows:**
```powershell
$env:ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

**Linux/Mac:**
```bash
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

**Or update the .env file:**
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### 3. Test the Integration
```bash
# Start the server
python main.py

# Test US stock (Alpha Vantage)
curl "http://127.0.0.1:8001/technical-analysis/MSFT"

# Test NSE stock (Demo data)
curl "http://127.0.0.1:8001/technical-analysis/TCS"
```

## API Endpoints

### Technical Analysis
```
GET /technical-analysis/{symbol}?period=3mo
```

**Examples:**
- US Stock: `/technical-analysis/MSFT`
- NSE Stock: `/technical-analysis/TCS`

### Chat Interface
The chat interface automatically detects technical analysis queries:
- "Show me technical analysis for Microsoft"
- "What's the RSI for Apple?"
- "Give me MACD signals for TCS"

## Data Sources

### Alpha Vantage (US Stocks)
- **Coverage**: US stocks (NASDAQ, NYSE, AMEX)
- **Indicators**: Professional RSI, MACD calculations
- **Rate Limits**: 25 requests/day (free), 500+/day (paid)
- **Quality**: High-quality, real-time data

### Demo Data (NSE Stocks)
- **Coverage**: Major NSE stocks (TCS, RELIANCE, INFY, etc.)
- **Purpose**: Demonstration and testing
- **Generation**: Realistic price movements with technical patterns
- **Note**: Not for actual trading decisions

## Supported Technical Indicators

### 📊 Alpha Vantage Indicators (US Stocks)
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **Real-time**: Calculated by Alpha Vantage servers

### 📈 Calculated Indicators (All Stocks)
- **SMA**: Simple Moving Averages (20, 50, 200)
- **EMA**: Exponential Moving Averages (12, 26)
- **Bollinger Bands**: Upper, middle, lower bands
- **Volume**: Volume analysis and ratios

## Rate Limits & Best Practices

### Free Tier Limits
- **25 requests per day**
- **5 API calls per minute**
- **Built-in delays** to respect limits

### Optimization
- **Caching**: Results are processed once per request
- **Fallback**: Demo data for unsupported markets
- **Error Handling**: Graceful degradation when API limits hit

## Example Responses

### US Stock (Alpha Vantage)
```json
{
  "symbol": "MSFT",
  "data_source": "Alpha Vantage",
  "indicators": {
    "rsi": 65.4,
    "macd": 2.15,
    "macd_signal": 1.89,
    "current_price": 378.85
  }
}
```

### NSE Stock (Demo)
```json
{
  "symbol": "TCS",
  "data_source": "Calculated",
  "indicators": {
    "rsi": 58.2,
    "macd": 12.45,
    "current_price": 3542.30
  }
}
```

## Troubleshooting

### Common Issues

**1. API Key Not Working**
- Verify key is correct
- Check environment variable is set
- Ensure account is activated

**2. Rate Limit Exceeded**
- Wait until next day (free tier)
- Consider upgrading to paid plan
- Use demo data for testing

**3. No Data for Symbol**
- US stocks: Check symbol exists on Alpha Vantage
- NSE stocks: Will use demo data automatically
- Try different symbol format

### Error Messages
- `"Alpha Vantage API limit"`: Rate limit hit
- `"No time series data found"`: Invalid symbol or no data
- `"Error fetching historical data"`: Network or API issue

## Migration from Yahoo Finance

### Changes Made
1. **Removed**: yfinance dependency
2. **Added**: Alpha Vantage API integration
3. **Enhanced**: Error handling and fallbacks
4. **Improved**: Professional-grade indicators

### Benefits
- **Reliability**: More stable than Yahoo Finance
- **Accuracy**: Professional financial data provider
- **Features**: Advanced technical indicators
- **Support**: Official API with documentation

## Next Steps

### Potential Enhancements
1. **Caching Layer**: Redis for API response caching
2. **More Markets**: European, Asian market support
3. **Real-time Updates**: WebSocket integration
4. **Advanced Indicators**: Stochastic, Williams %R, etc.

---

*For support with Alpha Vantage integration, check the [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/) or contact support.*
