# technical_analysis.py
"""
Technical analysis utilities and calculations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from config import RSI_PERIOD, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD, BOLLINGER_BANDS_PERIOD, BOLLINGER_BANDS_STD

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

import pandas_ta as ta


def get_stock_historical_data(symbol: str, period: str = "3mo", interval: str = "1d"):
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
        indicators['ema_12'] = df['Close'].ewm(span=MACD_FAST_PERIOD).mean().iloc[-1]
        indicators['ema_26'] = df['Close'].ewm(span=MACD_SLOW_PERIOD).mean().iloc[-1]
        
        # RSI (Relative Strength Index)
        if TALIB_AVAILABLE:
            rsi = talib.RSI(df['Close'].values, timeperiod=RSI_PERIOD)
            indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else None
        else:
            # Calculate RSI manually using pandas-ta
            rsi_series = ta.rsi(df['Close'], length=RSI_PERIOD)
            indicators['rsi'] = rsi_series.iloc[-1] if not rsi_series.empty else None
        
        # MACD (Moving Average Convergence Divergence)
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
