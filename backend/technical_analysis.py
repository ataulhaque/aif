# technical_analysis.py
"""
Technical analysis utilities and calculations using Alpha Vantage API
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from config import (
    RSI_PERIOD, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD, 
    BOLLINGER_BANDS_PERIOD, BOLLINGER_BANDS_STD, ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL
)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

import pandas_ta as ta


def get_stock_historical_data(symbol: str, period: str = "3mo", interval: str = "1d"):
    """
    Fetch historical stock data for technical analysis using Alpha Vantage API
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS', 'TCS.NS', 'MSFT')
        period: Data period (not used with Alpha Vantage - returns full history)
        interval: Data interval - supported: 1d (daily), 1wk (weekly), 1mo (monthly)
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        # Map interval to Alpha Vantage function
        if interval in ['1d', 'daily']:
            function = 'TIME_SERIES_DAILY'
        elif interval in ['1wk', 'weekly']:
            function = 'TIME_SERIES_WEEKLY'
        elif interval in ['1mo', 'monthly']:
            function = 'TIME_SERIES_MONTHLY'
        else:
            # Default to daily for unsupported intervals
            function = 'TIME_SERIES_DAILY'
          # Handle NSE stocks - remove .NS suffix for Alpha Vantage
        if symbol.endswith('.NS'):
            # For NSE stocks, we'll use demo data
            symbol_clean = symbol.replace('.NS', '')
            print(f"Using demo data for NSE stock: {symbol_clean}")
            return _get_demo_stock_data(symbol_clean)
        elif symbol in ['TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'SBIN', 'ICICIBANK', 'BHARTIARTL', 'ITC', 'WIPRO', 'HCLTECH']:
            # These are NSE symbols without .NS suffix
            print(f"Using demo data for NSE stock: {symbol}")
            return _get_demo_stock_data(symbol)
          # Prepare API request
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"Alpha Vantage API error: {data['Error Message']} - falling back to demo data")
                return _get_demo_stock_data(symbol)
                
            if 'Note' in data:
                print(f"Alpha Vantage API limit: {data['Note']} - falling back to demo data")
                return _get_demo_stock_data(symbol)
            
            # Check if we got valid data
            if 'Information' in data:
                print(f"Alpha Vantage rate limit hit - falling back to demo data for {symbol}")
                return _get_demo_stock_data(symbol)
                
        except Exception as e:
            print(f"Alpha Vantage API request failed: {e} - falling back to demo data")
            return _get_demo_stock_data(symbol)
          # Extract time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
                
        if not time_series_key:
            print(f"No time series data found in Alpha Vantage response for {symbol} - using demo data")
            return _get_demo_stock_data(symbol)
            
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'Date': datetime.strptime(date_str, '%Y-%m-%d'),
                'Open': float(values.get('1. open', 0)),
                'High': float(values.get('2. high', 0)),
                'Low': float(values.get('3. low', 0)),
                'Close': float(values.get('4. close', 0)),
                'Volume': int(float(values.get('5. volume', 0)))
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Filter data based on period
        if period and period != 'max':
            days_map = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, 
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
            }
            if period in days_map:
                cutoff_date = datetime.now() - timedelta(days=days_map[period])
                df = df[df.index >= cutoff_date]
        
        if df.empty:
            print(f"No historical data available for {symbol}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None


def _get_demo_stock_data(symbol: str) -> pd.DataFrame:
    """
    Generate demo stock data for NSE stocks when Alpha Vantage doesn't have coverage
    This is for demonstration purposes - in production, you'd use a proper NSE data source
    """
    try:
        # Generate 90 days of realistic demo data
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        # Start with a base price (varies by symbol)
        base_prices = {
            'TCS': 3500, 'RELIANCE': 2450, 'INFY': 1650, 'HDFCBANK': 1580,
            'SBIN': 750, 'ICICIBANK': 1150, 'BHARTIARTL': 950, 'ITC': 450
        }
        base_price = base_prices.get(symbol, 1000)
          # Generate realistic price movements
        np.random.seed(hash(symbol) % 1000)  # Consistent seed for each symbol
        returns = np.random.normal(0.001, 0.02, len(dates))  # Small daily returns with volatility
        prices = [base_price]
        
        for i, ret in enumerate(returns[1:], 1):
            prices.append(prices[-1] * (1 + ret))
          # Generate OHLCV data
        df_data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = 0.015  # 1.5% intraday volatility
            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            open_price = prices[i-1] if i > 0 else close
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (higher volume on price moves)
            base_volume = 1000000
            ret = returns[i] if i < len(returns) else 0
            volume_multiplier = 1 + abs(ret) * 10  # Higher volume on bigger moves
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2))
            df_data.append({
                'Open': float(open_price),
                'High': float(high),
                'Low': float(low),
                'Close': float(close),
                'Volume': int(volume)
            })
        
        df = pd.DataFrame(df_data, index=dates)
        print(f"Generated demo data for {symbol} with {len(df)} days")
        return df
        
    except Exception as e:
        print(f"Error generating demo data for {symbol}: {e}")
        return None


def calculate_technical_indicators(df, symbol=None):
    """
    Calculate various technical indicators for stock data
    
    Args:
        df: pandas DataFrame with OHLCV data
        symbol: Stock symbol (optional, for Alpha Vantage indicators)
        
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
        
        # Check for NaN values and replace with None
        for key in ['sma_20', 'sma_50', 'sma_200']:
            if key in indicators and pd.isna(indicators[key]):
                indicators[key] = None
          # Exponential Moving Averages
        indicators['ema_12'] = df['Close'].ewm(span=MACD_FAST_PERIOD).mean().iloc[-1]
        indicators['ema_26'] = df['Close'].ewm(span=MACD_SLOW_PERIOD).mean().iloc[-1]
        
        # Check for NaN values and replace with None
        for key in ['ema_12', 'ema_26']:
            if key in indicators and pd.isna(indicators[key]):
                indicators[key] = None
        
        # Try to get Alpha Vantage technical indicators first (for US stocks)
        alpha_indicators = {}
        if symbol and not symbol.endswith('.NS') and ALPHA_VANTAGE_API_KEY != "demo":
            alpha_indicators = get_alpha_vantage_technical_indicators(symbol)
        
        # RSI (Relative Strength Index)
        if 'alpha_vantage_rsi' in alpha_indicators:
            indicators['rsi'] = alpha_indicators['alpha_vantage_rsi']
        else:
            if TALIB_AVAILABLE:
                rsi = talib.RSI(df['Close'].values, timeperiod=RSI_PERIOD)
                indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else None
            else:
                # Calculate RSI manually using pandas-ta
                rsi_series = ta.rsi(df['Close'], length=RSI_PERIOD)
                indicators['rsi'] = rsi_series.iloc[-1] if not rsi_series.empty else None
        
        # MACD (Moving Average Convergence Divergence)
        if all(key in alpha_indicators for key in ['alpha_vantage_macd', 'alpha_vantage_macd_signal', 'alpha_vantage_macd_hist']):
            indicators['macd'] = alpha_indicators['alpha_vantage_macd']
            indicators['macd_signal'] = alpha_indicators['alpha_vantage_macd_signal']
            indicators['macd_histogram'] = alpha_indicators['alpha_vantage_macd_hist']
        else:
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(df['Close'].values, 
                                                         fastperiod=MACD_FAST_PERIOD, 
                                                         slowperiod=MACD_SLOW_PERIOD, 
                                                         signalperiod=MACD_SIGNAL_PERIOD)
                indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else None
                indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else None
                indicators['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else None
            else:
                # Calculate MACD manually
                ema12 = df['Close'].ewm(span=MACD_FAST_PERIOD).mean()
                ema26 = df['Close'].ewm(span=MACD_SLOW_PERIOD).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD).mean()
                histogram = macd_line - signal_line
                
                indicators['macd'] = macd_line.iloc[-1]
                indicators['macd_signal'] = signal_line.iloc[-1]
                indicators['macd_histogram'] = histogram.iloc[-1]
          # Bollinger Bands
        bb_period = BOLLINGER_BANDS_PERIOD
        bb_std = BOLLINGER_BANDS_STD
        sma_bb = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        indicators['bb_upper'] = (sma_bb + bb_std * bb_std_dev).iloc[-1]
        indicators['bb_lower'] = (sma_bb - bb_std * bb_std_dev).iloc[-1]
        indicators['bb_middle'] = sma_bb.iloc[-1]
        
        # Check for NaN values in Bollinger Bands
        for key in ['bb_upper', 'bb_lower', 'bb_middle']:
            if key in indicators and pd.isna(indicators[key]):
                indicators[key] = None
          # Current price
        indicators['current_price'] = float(df['Close'].iloc[-1])
        indicators['previous_close'] = float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Close'].iloc[-1])
        
        # Volume indicators (convert to Python int)
        indicators['volume_sma_10'] = float(df['Volume'].rolling(window=10).mean().iloc[-1])
        indicators['current_volume'] = int(df['Volume'].iloc[-1])
        
        # Convert all numpy types to Python types for JSON serialization
        for key, value in indicators.items():
            if hasattr(value, 'item'):  # numpy scalar
                indicators[key] = value.item()
            elif pd.isna(value):
                indicators[key] = None
        
        # Add data source information
        indicators['data_source'] = 'Alpha Vantage' if alpha_indicators else 'Calculated'
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return None


def get_alpha_vantage_technical_indicators(symbol: str):
    """
    Get technical indicators directly from Alpha Vantage API
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with Alpha Vantage technical indicators
    """
    indicators = {}
    
    try:
        # RSI from Alpha Vantage
        rsi_params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': RSI_PERIOD,
            'series_type': 'close',
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        rsi_response = requests.get(ALPHA_VANTAGE_BASE_URL, params=rsi_params, timeout=30)
        if rsi_response.status_code == 200:
            rsi_data = rsi_response.json()
            if 'Technical Analysis: RSI' in rsi_data:
                rsi_values = rsi_data['Technical Analysis: RSI']
                latest_date = max(rsi_values.keys())
                indicators['alpha_vantage_rsi'] = float(rsi_values[latest_date]['RSI'])
        
        # Add a small delay to respect API limits
        time.sleep(0.2)
        
        # MACD from Alpha Vantage
        macd_params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': 'daily',
            'series_type': 'close',
            'fastperiod': MACD_FAST_PERIOD,
            'slowperiod': MACD_SLOW_PERIOD,
            'signalperiod': MACD_SIGNAL_PERIOD,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        macd_response = requests.get(ALPHA_VANTAGE_BASE_URL, params=macd_params, timeout=30)
        if macd_response.status_code == 200:
            macd_data = macd_response.json()
            if 'Technical Analysis: MACD' in macd_data:
                macd_values = macd_data['Technical Analysis: MACD']
                latest_date = max(macd_values.keys())
                indicators['alpha_vantage_macd'] = float(macd_values[latest_date]['MACD'])
                indicators['alpha_vantage_macd_signal'] = float(macd_values[latest_date]['MACD_Signal'])
                indicators['alpha_vantage_macd_hist'] = float(macd_values[latest_date]['MACD_Hist'])
        
    except Exception as e:
        print(f"Error fetching Alpha Vantage technical indicators: {e}")
    
    return indicators


def generate_trading_signals(indicators: dict, symbol: str) -> dict:
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
