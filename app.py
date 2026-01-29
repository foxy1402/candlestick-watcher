import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datetime import datetime, timedelta
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Pattern Watcher", page_icon="🕯️")

# --- Session State Initialization ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['BTC-USD', 'ETH-USD', 'SOL-USD']
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'yahoo'  # Default to Yahoo Finance

# --- Coinalyze API Helper Functions ---
def get_coinalyze_api_key() -> str:
    """Get Coinalyze API key from environment variable."""
    return os.environ.get('COINALYZE_API_KEY', '')

# --- CryptoCompare API Helper Functions ---
def get_cryptocompare_api_key() -> str:
    """Get CryptoCompare API key from Streamlit secrets or environment variable."""
    try:
        return st.secrets.get('CRYPTOCOMPARE_API_KEY', '')
    except:
        pass
    return os.environ.get('CRYPTOCOMPARE_API_KEY', '')

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data_cryptocompare(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from CryptoCompare API (primary data source)."""
    try:
        api_key = get_cryptocompare_api_key()
        if not api_key:
            return pd.DataFrame()
        
        if '-' in symbol:
            parts = symbol.split('-')
            fsym = parts[0].upper()
            tsym = parts[1].upper() if len(parts) > 1 else 'USD'
        else:
            fsym = symbol.replace('USDT', '').replace('USD', '').upper()
            tsym = 'USD'
        
        aggregate = 7 if interval == '1wk' else 1
        
        url = "https://min-api.cryptocompare.com/data/histoday"
        params = {
            'fsym': fsym,
            'tsym': tsym,
            'allData': 'true',
            'aggregate': aggregate,
            'api_key': api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('Response') != 'Success' or 'Data' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['Data'])
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumeto'].astype(float)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[(df['open'] > 0) & (df['close'] > 0)]
        return df
    except Exception:
        return pd.DataFrame()

def fetch_data_cryptocompare_raw(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch from CryptoCompare WITHOUT cache - for parallel threads."""
    try:
        api_key = get_cryptocompare_api_key()
        if not api_key:
            return pd.DataFrame()
        
        if '-' in symbol:
            parts = symbol.split('-')
            fsym = parts[0].upper()
            tsym = parts[1].upper() if len(parts) > 1 else 'USD'
        else:
            fsym = symbol.replace('USDT', '').replace('USD', '').upper()
            tsym = 'USD'
        
        aggregate = 7 if interval == '1wk' else 1
        
        url = "https://min-api.cryptocompare.com/data/histoday"
        params = {
            'fsym': fsym,
            'tsym': tsym,
            'allData': 'true',
            'aggregate': aggregate,
            'api_key': api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('Response') != 'Success' or 'Data' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['Data'])
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumeto'].astype(float)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[(df['open'] > 0) & (df['close'] > 0)]
        return df
    except Exception:
        return pd.DataFrame()


def convert_symbol_to_coinalyze(symbol: str) -> str:
    """Convert Yahoo Finance symbol to Coinalyze format.
    BTC-USD -> BTCUSD_PERP.A (Binance perpetual)
    """
    base = symbol.upper().replace('-USD', '').replace('USDT', '')
    return f"{base}USD_PERP.A"  # Binance perpetual format

@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_interest(symbol: str) -> dict:
    """Fetch current Open Interest from Coinalyze API."""
    try:
        api_key = get_coinalyze_api_key()
        if not api_key:
            return {'error': 'COINALYZE_API_KEY not set', 'oi': 0, 'symbol': symbol}
        
        coinalyze_symbol = convert_symbol_to_coinalyze(symbol)
        
        url = f"https://api.coinalyze.net/v1/open-interest"
        params = {'symbols': coinalyze_symbol}
        headers = {'api_key': api_key}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return {'error': f'API error: {response.status_code}', 'oi': 0, 'symbol': symbol}
        
        data = response.json()
        if data and len(data) > 0:
            # Coinalyze returns 'value' for current OI, not 'openInterest'
            current_oi = float(data[0].get('value', 0))
            return {
                'symbol': symbol,
                'coinalyze_symbol': coinalyze_symbol,
                'oi': current_oi,
                'timestamp': datetime.now().isoformat(),
                'error': None
            }
        return {'error': 'No data', 'oi': 0, 'symbol': symbol}
    except Exception as e:
        return {'error': str(e), 'oi': 0, 'symbol': symbol}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_interest_history(symbol: str, period: str = 'daily', days: int = 365) -> pd.DataFrame:
    """Fetch historical Open Interest data from Coinalyze API.
    
    Coinalyze supports unlimited history!
    Intervals: minute, 5minute, 15minute, 30minute, hour, 2hour, 4hour, 6hour, 12hour, daily, weekly
    """
    try:
        api_key = get_coinalyze_api_key()
        if not api_key:
            return pd.DataFrame()
        
        coinalyze_symbol = convert_symbol_to_coinalyze(symbol)
        
        # Calculate time range
        to_time = int(datetime.now().timestamp())
        from_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        url = f"https://api.coinalyze.net/v1/open-interest-history"
        params = {
            'symbols': coinalyze_symbol,
            'interval': period,
            'from': from_time,
            'to': to_time
        }
        headers = {'api_key': api_key}
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        if not data or len(data) == 0 or 'history' not in data[0]:
            return pd.DataFrame()
        
        history = data[0]['history']
        df = pd.DataFrame(history)
        
        # Coinalyze returns OHLC format for OI: t=timestamp, o=open, h=high, l=low, c=close
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        df['sumOpenInterest'] = df['c'].astype(float)  # Use close value for current OI
        
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rate(symbol: str) -> dict:
    """Fetch current funding rate from Coinalyze API."""
    try:
        api_key = get_coinalyze_api_key()
        if not api_key:
            return {'error': 'API key not set', 'rate': 0}
        
        coinalyze_symbol = convert_symbol_to_coinalyze(symbol)
        
        url = f"https://api.coinalyze.net/v1/funding-rate"
        params = {'symbols': coinalyze_symbol}
        headers = {'api_key': api_key}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return {'error': f'API {response.status_code}', 'rate': 0}
        
        data = response.json()
        if data and len(data) > 0:
            # Coinalyze returns 'value' for funding rate (already in percentage form 0.01 = 0.01%)
            rate = float(data[0].get('value', 0))
            return {
                'rate': rate,
                'timestamp': data[0].get('update'),
                'error': None
            }
        return {'rate': 0, 'error': 'No data'}
    except Exception as e:
        return {'rate': 0, 'error': str(e)}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_long_short_ratio(symbol: str) -> dict:
    """Fetch long/short ratio from Coinalyze API."""
    try:
        api_key = get_coinalyze_api_key()
        if not api_key:
            return {'error': 'API key not set', 'ratio': 1.0, 'long_pct': 50, 'short_pct': 50}
        
        coinalyze_symbol = convert_symbol_to_coinalyze(symbol)
        
        # Get recent L/S ratio history
        to_time = int(datetime.now().timestamp())
        from_time = int((datetime.now() - timedelta(hours=24)).timestamp())
        
        url = f"https://api.coinalyze.net/v1/long-short-ratio-history"
        params = {
            'symbols': coinalyze_symbol,
            'interval': '1hour',  # Must be '1hour' not 'hour'
            'from': from_time,
            'to': to_time
        }
        headers = {'api_key': api_key}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return {'error': f'API {response.status_code}', 'ratio': 1.0, 'long_pct': 50, 'short_pct': 50}
        
        data = response.json()
        if data and len(data) > 0 and 'history' in data[0] and len(data[0]['history']) > 0:
            # Coinalyze returns: r=ratio, l=long%, s=short% directly
            latest = data[0]['history'][-1]
            ratio = float(latest.get('r', 1.0))
            long_pct = float(latest.get('l', 50))
            short_pct = float(latest.get('s', 50))
            
            return {
                'ratio': ratio,
                'long_pct': long_pct,
                'short_pct': short_pct,
                'error': None
            }
        return {'ratio': 1.0, 'long_pct': 50, 'short_pct': 50, 'error': 'No data'}
    except Exception as e:
        return {'ratio': 1.0, 'long_pct': 50, 'short_pct': 50, 'error': str(e)}


def analyze_oi_advisory(oi_change_pct: float, price_change_pct: float, 
                       oi_history: pd.DataFrame = None) -> dict:
    """
    Generate investment advisory based on OI + Price divergence.
    Uses statistical thresholds to avoid noise.
    
    Args:
        oi_change_pct: % change in open interest
        price_change_pct: % change in price
        oi_history: Historical OI data for calculating volatility-adjusted thresholds
    """
    
    # Calculate dynamic thresholds based on historical volatility if available
    if oi_history is not None and len(oi_history) > 20:
        oi_volatility = oi_history['sumOpenInterest'].pct_change().std()
        # Threshold = 1 standard deviation
        oi_threshold = max(1.0, oi_volatility * 100)  # At least 1%
    else:
        oi_threshold = 2.0  # Default 2% (more conservative than 0.5%)
    
    price_threshold = 1.5  # Price threshold (crypto is volatile)
    
    # Strong signals require both metrics above threshold
    if oi_change_pct > oi_threshold and price_change_pct > price_threshold:
        return {
            'signal': 'BULLISH',
            'emoji': '🟢',
            'label': 'STRONG UPTREND',
            'description': f'OI +{oi_change_pct:.1f}% and Price +{price_change_pct:.1f}% = New money entering',
            'advisory': '✅ Favorable for holding or adding positions',
            'color': 'green',
            'score': 85,
            'confidence': 'HIGH'
        }
    
    elif oi_change_pct > oi_threshold and price_change_pct < -price_threshold:
        return {
            'signal': 'BEARISH',
            'emoji': '🔴',
            'label': 'SHORT BUILDUP',
            'description': f'OI +{oi_change_pct:.1f}% while Price -{abs(price_change_pct):.1f}% = Shorts entering',
            'advisory': '⚠️ Caution - High risk of further downside',
            'color': 'red',
            'score': 20,
            'confidence': 'HIGH'
        }
    
    elif oi_change_pct < -oi_threshold and price_change_pct > price_threshold:
        return {
            'signal': 'WEAK_RALLY',
            'emoji': '🟡',
            'label': 'SHORT SQUEEZE',
            'description': f'OI -{abs(oi_change_pct):.1f}% while Price +{price_change_pct:.1f}% = Short covering',
            'advisory': '⏳ Rally may be temporary - Wait for OI confirmation',
            'color': 'yellow',
            'score': 50,
            'confidence': 'MEDIUM'
        }
    
    elif oi_change_pct < -oi_threshold and price_change_pct < -price_threshold:
        return {
            'signal': 'CAPITULATION',
            'emoji': '🟠',
            'label': 'LIQUIDATION CASCADE',
            'description': f'OI -{abs(oi_change_pct):.1f}% and Price -{abs(price_change_pct):.1f}% = Mass liquidations',
            'advisory': '👀 Potential bottom forming - Watch for reversal with OI stabilization',
            'color': 'orange',
            'score': 40,
            'confidence': 'MEDIUM'
        }
    
    # Weak signals (below threshold) = noise
    else:
        magnitude = 'small' if abs(oi_change_pct) < oi_threshold/2 else 'moderate'
        return {
            'signal': 'NEUTRAL',
            'emoji': '⚪',
            'label': 'NO CLEAR SIGNAL',
            'description': f'Changes too {magnitude} to be significant (OI {oi_change_pct:+.1f}%, Price {price_change_pct:+.1f}%)',
            'advisory': '⏸️ Wait for clearer signals above threshold',
            'color': 'gray',
            'score': 50,
            'confidence': 'LOW'
        }

def calculate_derivatives_score(oi_change: float, funding_rate: float, long_short_ratio: float, price_change: float) -> dict:
    """
    Calculate comprehensive derivatives-based investment score (0-100).
    Combines OI, funding rate, and positioning data for actionable advice.
    """
    score = 50  # Start neutral
    factors = []
    
    # OI Trend (+/- 20 points)
    if oi_change > 1:
        score += 15
        factors.append(('OI Rising', '+15', 'green'))
    elif oi_change > 0:
        score += 5
        factors.append(('OI Slightly Up', '+5', 'green'))
    elif oi_change < -1:
        score -= 15
        factors.append(('OI Falling', '-15', 'red'))
    elif oi_change < 0:
        score -= 5
        factors.append(('OI Slightly Down', '-5', 'red'))
    
    # Funding Rate (+/- 15 points) - Negative funding = bullish for longs
    if funding_rate < -0.01:  # Very negative = shorts paying longs
        score += 15
        factors.append(('Funding Negative (Bullish)', '+15', 'green'))
    elif funding_rate < 0:
        score += 5
        factors.append(('Funding Slightly Negative', '+5', 'green'))
    elif funding_rate > 0.05:  # High positive = overheated longs
        score -= 15
        factors.append(('Funding High (Overheated)', '-15', 'red'))
    elif funding_rate > 0.01:
        score -= 5
        factors.append(('Funding Elevated', '-5', 'orange'))
    
    # Long/Short Ratio (+/- 15 points) - Contrarian indicator
    if long_short_ratio < 0.8:  # More shorts = potential squeeze
        score += 15
        factors.append(('Shorts Crowded (Squeeze Risk)', '+15', 'green'))
    elif long_short_ratio < 1.0:
        score += 5
        factors.append(('Slight Short Bias', '+5', 'green'))
    elif long_short_ratio > 1.5:  # Too many longs = risk
        score -= 15
        factors.append(('Longs Crowded (Risk)', '-15', 'red'))
    elif long_short_ratio > 1.2:
        score -= 5
        factors.append(('Long Bias', '-5', 'orange'))
    
    # Price trend alignment (+/- 10 points)
    if oi_change > 0 and price_change > 0:
        score += 10
        factors.append(('Trend Confirmed', '+10', 'green'))
    elif oi_change > 0 and price_change < 0:
        score -= 10
        factors.append(('Bearish Divergence', '-10', 'red'))
    
    # Clamp score
    score = max(0, min(100, score))
    
    # Generate recommendation
    if score >= 70:
        recommendation = '🟢 FAVORABLE - Good conditions for long positions'
    elif score >= 55:
        recommendation = '🟡 NEUTRAL-BULLISH - Cautiously optimistic'
    elif score >= 45:
        recommendation = '⚪ NEUTRAL - Wait for clearer signals'
    elif score >= 30:
        recommendation = '🟠 CAUTION - Unfavorable conditions'
    else:
        recommendation = '🔴 AVOID - High-risk environment'
    
    return {
        'score': score,
        'factors': factors,
        'recommendation': recommendation
    }

# --- Helper Functions for Data Safety ---
def safe_pct_change(current: float, previous: float) -> float:
    """Calculate percentage change with comprehensive safety checks."""
    if previous == 0 or pd.isna(previous) or pd.isna(current) or np.isinf(previous) or np.isinf(current):
        return 0.0
    # Additional check for very small denominators to avoid extreme percentages
    if abs(previous) < 1e-10:
        return 0.0
    return ((current - previous) / previous) * 100

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero/NaN protection."""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator

def safe_series_diff(series: pd.Series, lookback: int) -> float:
    """Safely calculate series difference with lookback."""
    if len(series) < lookback + 1:
        return 0.0
    try:
        recent = series.iloc[-1]
        past = series.iloc[-lookback]
        if pd.isna(recent) or pd.isna(past):
            return 0.0
        return recent - past
    except (IndexError, KeyError):
        return 0.0

# --- Constants ---
PATTERN_RANKINGS = {
    "CDL3LINESTRIKE_Bull": 1, "CDL3LINESTRIKE_Bear": 2, "CDL3BLACKCROWS_Bull": 3, "CDL3BLACKCROWS_Bear": 3,
    "CDLEVENINGSTAR_Bull": 4, "CDLEVENINGSTAR_Bear": 4, "CDLTASUKIGAP_Bull": 5, "CDLTASUKIGAP_Bear": 5,
    "CDLINVERTEDHAMMER_Bull": 6, "CDLINVERTEDHAMMER_Bear": 6, "CDLMATCHINGLOW_Bull": 7, "CDLMATCHINGLOW_Bear": 7,
    "CDLABANDONEDBABY_Bull": 8, "CDLABANDONEDBABY_Bear": 8, "CDLBREAKAWAY_Bull": 10, "CDLBREAKAWAY_Bear": 10,
    "CDLMORNINGSTAR_Bull": 12, "CDLMORNINGSTAR_Bear": 12, "CDLPIERCING_Bull": 13, "CDLPIERCING_Bear": 13,
    "CDLSTICKSANDWICH_Bull": 14, "CDLSTICKSANDWICH_Bear": 14, "CDLTHRUSTING_Bull": 15, "CDLTHRUSTING_Bear": 15,
    "CDLINNECK_Bull": 17, "CDLINNECK_Bear": 17, "CDL3INSIDE_Bull": 20, "CDL3INSIDE_Bear": 56,
    "CDLHOMINGPIGEON_Bull": 21, "CDLHOMINGPIGEON_Bear": 21, "CDLDARKCLOUDCOVER_Bull": 22, "CDLDARKCLOUDCOVER_Bear": 22,
    "CDLIDENTICAL3CROWS_Bull": 24, "CDLIDENTICAL3CROWS_Bear": 24, "CDLMORNINGDOJISTAR_Bull": 25, "CDLMORNINGDOJISTAR_Bear": 25,
    "CDLXSIDEGAP3METHODS_Bull": 27, "CDLXSIDEGAP3METHODS_Bear": 26, "CDLTRISTAR_Bull": 28, "CDLTRISTAR_Bear": 76,
    "CDLGAPSIDESIDEWHITE_Bull": 46, "CDLGAPSIDESIDEWHITE_Bear": 29, "CDLEVENINGDOJISTAR_Bull": 30, "CDLEVENINGDOJISTAR_Bear": 30,
    "CDL3WHITESOLDIERS_Bull": 32, "CDL3WHITESOLDIERS_Bear": 32, "CDLONNECK_Bull": 33, "CDLONNECK_Bear": 33,
    "CDL3OUTSIDE_Bull": 34, "CDL3OUTSIDE_Bear": 39, "CDLRICKSHAWMAN_Bull": 35, "CDLRICKSHAWMAN_Bear": 35,
    "CDLSEPARATINGLINES_Bull": 36, "CDLSEPARATINGLINES_Bear": 40, "CDLLONGLEGGEDDOJI_Bull": 37, "CDLLONGLEGGEDDOJI_Bear": 37,
    "CDLHARAMI_Bull": 38, "CDLHARAMI_Bear": 72, "CDLLADDERBOTTOM_Bull": 41, "CDLLADDERBOTTOM_Bear": 41,
    "CDLCLOSINGMARUBOZU_Bull": 70, "CDLCLOSINGMARUBOZU_Bear": 43, "CDLTAKURI_Bull": 47, "CDLTAKURI_Bear": 47,
    "CDLDOJISTAR_Bull": 49, "CDLDOJISTAR_Bear": 51, "CDLHARAMICROSS_Bull": 50, "CDLHARAMICROSS_Bear": 80,
    "CDLADVANCEBLOCK_Bull": 54, "CDLADVANCEBLOCK_Bear": 54, "CDLSHOOTINGSTAR_Bull": 55, "CDLSHOOTINGSTAR_Bear": 55,
    "CDLMARUBOZU_Bull": 71, "CDLMARUBOZU_Bear": 57, "CDLUNIQUE3RIVER_Bull": 60, "CDLUNIQUE3RIVER_Bear": 60,
    "CDL2CROWS_Bull": 61, "CDL2CROWS_Bear": 61, "CDLBELTHOLD_Bull": 62, "CDLBELTHOLD_Bear": 63,
    "CDLHAMMER_Bull": 65, "CDLHAMMER_Bear": 65, "CDLHIGHWAVE_Bull": 67, "CDLHIGHWAVE_Bear": 67,
    "CDLSPINNINGTOP_Bull": 69, "CDLSPINNINGTOP_Bear": 73, "CDLUPSIDEGAP2CROWS_Bull": 74, "CDLUPSIDEGAP2CROWS_Bear": 74,
    "CDLGRAVESTONEDOJI_Bull": 77, "CDLGRAVESTONEDOJI_Bear": 77, "CDLHIKKAKEMOD_Bull": 82, "CDLHIKKAKEMOD_Bear": 81,
    "CDLHIKKAKE_Bull": 85, "CDLHIKKAKE_Bear": 83, "CDLENGULFING_Bull": 84, "CDLENGULFING_Bear": 91,
    "CDLMATHOLD_Bull": 86, "CDLMATHOLD_Bear": 86, "CDLHANGINGMAN_Bull": 87, "CDLHANGINGMAN_Bear": 87,
    "CDLRISEFALL3METHODS_Bull": 94, "CDLRISEFALL3METHODS_Bear": 89, "CDLKICKING_Bull": 96, "CDLKICKING_Bear": 102,
    "CDLDRAGONFLYDOJI_Bull": 98, "CDLDRAGONFLYDOJI_Bear": 98, "CDLCONCEALBABYSWALL_Bull": 101, "CDLCONCEALBABYSWALL_Bear": 101,
    "CDL3STARSINSOUTH_Bull": 103, "CDL3STARSINSOUTH_Bear": 103, "CDLDOJI_Bull": 104, "CDLDOJI_Bear": 104
}

# --- Optimized Functions ---

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data_yfinance(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance (yfinance)."""
    try:
        ticker = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        df = yf.download(ticker, period="max", interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_data_yfinance_raw(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch from yfinance WITHOUT cache - for parallel threads."""
    try:
        ticker = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        df = yf.download(ticker, period="max", interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, interval: str, data_source: str = 'yahoo') -> pd.DataFrame:
    """
    Fetch OHLCV data based on selected data source.
    Cache key includes data_source automatically.
    """
    fetch_start = datetime.now()
    
    if data_source == 'cryptocompare':
        df = fetch_data_cryptocompare(symbol, interval)
        if df.empty:
            st.warning(f"⚠️ CryptoCompare returned no data for {symbol}, trying Yahoo Finance...")
            df = fetch_data_yfinance(symbol, interval)
            if not df.empty:
                st.info(f"✅ Loaded from Yahoo Finance (fallback)")
        return df
    else:
        df = fetch_data_yfinance(symbol, interval)
        if df.empty:
            st.warning(f"⚠️ Yahoo Finance returned no data for {symbol}, trying CryptoCompare...")
            df = fetch_data_cryptocompare(symbol, interval)
            if not df.empty:
                st.info(f"✅ Loaded from CryptoCompare (fallback)")
        return df


def fetch_data_raw(symbol: str, interval: str, data_source: str = 'yahoo') -> pd.DataFrame:
    """Fetch data WITHOUT cache based on selected data source."""
    if data_source == 'cryptocompare':
        # Try CryptoCompare first, fallback to yfinance
        df = fetch_data_cryptocompare_raw(symbol, interval)
        if not df.empty:
            return df
        return fetch_data_yfinance_raw(symbol, interval)
    else:
        # Yahoo Finance (default), fallback to CryptoCompare
        df = fetch_data_yfinance_raw(symbol, interval)
        if not df.empty:
            return df
        return fetch_data_cryptocompare_raw(symbol, interval)


def detect_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized pattern detection - much faster than iterrows."""
    candle_names = talib.get_function_groups()['Pattern Recognition']
    op, hi, lo, cl = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    # Apply all patterns at once (vectorized)
    pattern_results = {}
    for candle in candle_names:
        pattern_results[candle] = getattr(talib, candle)(op, hi, lo, cl)
    
    # Initialize result columns
    df['candlestick_pattern'] = "NO_PATTERN"
    df['candlestick_match_count'] = 0
    df['pattern_direction'] = 'neutral'
    
    # Vectorized pattern detection
    for i in range(len(df)):
        found = []
        for candle, values in pattern_results.items():
            if values[i] != 0:
                direction = 'Bull' if values[i] > 0 else 'Bear'
                pattern_key = f"{candle}_{direction}"
                rank = PATTERN_RANKINGS.get(pattern_key, 999)
                found.append((pattern_key, rank, values[i]))
        
        if found:
            found.sort(key=lambda x: x[1])
            df.iloc[i, df.columns.get_loc('candlestick_pattern')] = found[0][0]
            df.iloc[i, df.columns.get_loc('candlestick_match_count')] = len(found)
            df.iloc[i, df.columns.get_loc('pattern_direction')] = 'bullish' if found[0][2] > 0 else 'bearish'
    
    df['pattern_display'] = df['candlestick_pattern'].str.replace('NO_PATTERN|CDL|_Bull|_Bear', '', regex=True)
    return df

def analyze_ad_phase_fast(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """Optimized A/D analysis using numpy vectorization."""
    # Vectorized A/D calculation with NaN handling
    df['ad'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)
    df['ad'] = df['ad'].fillna(0)  # Fill NaN values
    df['ad_ema'] = talib.EMA(df['ad'].values, timeperiod=21)
    df['ad_ema'] = df['ad_ema'].fillna(df['ad'])  # Use raw AD where EMA is NaN
    
    # Vectorized phase detection
    df['price_change'] = df['close'].diff(lookback)
    df['ad_change'] = df['ad'].diff(lookback)
    
    # Use numpy select for vectorized conditions
    conditions = [
        (df['price_change'] < 0) & (df['ad_change'] > 0),
        (df['price_change'] > 0) & (df['ad_change'] < 0),
        (df['ad_change'] > 0),
        (df['ad_change'] < 0),
    ]
    choices = ['accumulation', 'distribution', 'uptrend', 'downtrend']
    df['phase'] = np.select(conditions, choices, default='neutral')
    
    # Current phase
    if len(df) >= lookback:
        recent = df.tail(lookback)
        price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
        ad_change = recent['ad'].iloc[-1] - recent['ad'].iloc[0]
        
        if price_change < 0 and ad_change > 0:
            return "accumulation", "green", df
        elif price_change > 0 and ad_change < 0:
            return "distribution", "red", df
        elif ad_change > 0:
            return "uptrend", "green", df
        elif ad_change < 0:
            return "downtrend", "red", df
    return "neutral", "gray", df

def detect_wyckoff_enhanced(df: pd.DataFrame, lookback: int = 52) -> dict:
    """
    Enhanced Wyckoff detection with statistical thresholds and volume confirmation.
    
    Returns:
        dict with phase, emoji, label, description, color, and confidence
    """
    if len(df) < lookback:
        return {
            "phase": "Insufficient Data", 
            "emoji": "⚪", 
            "label": "N/A", 
            "description": "Need more data", 
            "color": "gray",
            "confidence": "NONE"
        }
    
    recent = df.tail(lookback)
    current_price = recent['close'].iloc[-1]
    price_high = recent['high'].max()
    price_low = recent['low'].min()
    price_range = price_high - price_low
    
    if price_range == 0:
        return {
            "phase": "Ranging", 
            "emoji": "↔️", 
            "label": "SIDEWAYS", 
            "description": "Market consolidating", 
            "color": "gray",
            "confidence": "MEDIUM"
        }
    
    # Use quartiles instead of arbitrary 0.3/0.7
    q1 = recent['close'].quantile(0.25)
    q3 = recent['close'].quantile(0.75)
    
    price_in_lower_quartile = current_price <= q1
    price_in_upper_quartile = current_price >= q3
    
    # A/D trend
    ad_trend = recent['ad'].iloc[-1] - recent['ad'].iloc[0] if 'ad' in recent.columns else 0
    price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
    
    # Volume confirmation
    volume_avg = recent['volume'].mean()
    recent_volume = recent['volume'].tail(10).mean()
    volume_increasing = recent_volume > volume_avg * 1.1
    
    # Price volatility (for confidence)
    volatility = recent['close'].std() / recent['close'].mean()
    
    # Calculate confidence based on data quality
    confidence = "HIGH"
    if volatility > 0.15:  # High volatility
        confidence = "MEDIUM"
    if len(recent) < lookback * 0.8:  # Missing data
        confidence = "LOW"
    
    # Accumulation: Low price, A/D rising, volume confirmation
    if price_in_lower_quartile and ad_trend > 0:
        strength = "STRONG" if volume_increasing else "MODERATE"
        return {
            "phase": "Accumulation",
            "emoji": "🛒",
            "label": f"{strength} ACCUMULATION",
            "description": "Smart money accumulating at lower prices" + 
                          (" with volume confirmation" if volume_increasing else ""),
            "color": "green",
            "confidence": confidence
        }
    
    # Distribution: High price, A/D falling, volume confirmation
    elif price_in_upper_quartile and ad_trend < 0:
        strength = "STRONG" if volume_increasing else "MODERATE"
        return {
            "phase": "Distribution",
            "emoji": "💸",
            "label": f"{strength} DISTRIBUTION",
            "description": "Smart money distributing at higher prices" +
                          (" with volume confirmation" if volume_increasing else ""),
            "color": "red",
            "confidence": confidence
        }
    
    # Markup: A/D and price both rising
    elif ad_trend > 0 and price_trend > 0:
        return {
            "phase": "Markup",
            "emoji": "📈",
            "label": "UPTREND",
            "description": "Healthy uptrend with accumulation continuing",
            "color": "green",
            "confidence": confidence
        }
    
    # Markdown: A/D and price both falling
    elif ad_trend < 0 and price_trend < 0:
        return {
            "phase": "Markdown",
            "emoji": "📉",
            "label": "DOWNTREND",
            "description": "Distribution continuing, downtrend confirmed",
            "color": "red",
            "confidence": confidence
        }
    
    # Default: Ranging
    return {
        "phase": "Ranging",
        "emoji": "↔️",
        "label": "CONSOLIDATION",
        "description": "Market consolidating, wait for clear direction",
        "color": "gray",
        "confidence": confidence
    }

def generate_signals_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized signal generation."""
    df['signal'] = 'none'
    df['signal_strength'] = 'none'
    
    # Vectorized conditions
    has_pattern = df['candlestick_pattern'] != 'NO_PATTERN'
    is_bullish = df['pattern_direction'] == 'bullish'
    is_bearish = df['pattern_direction'] == 'bearish'
    is_accum = df['phase'] == 'accumulation'
    is_distrib = df['phase'] == 'distribution'
    is_uptrend = df['phase'] == 'uptrend'
    is_downtrend = df['phase'] == 'downtrend'
    
    df.loc[has_pattern & is_accum & is_bullish, 'signal'] = 'strong_buy'
    df.loc[has_pattern & is_accum & is_bullish, 'signal_strength'] = 'STRONG BUY ⭐🟢'
    df.loc[has_pattern & is_distrib & is_bearish, 'signal'] = 'strong_sell'
    df.loc[has_pattern & is_distrib & is_bearish, 'signal_strength'] = 'STRONG SELL ⭐🔴'
    df.loc[has_pattern & is_uptrend & is_bullish, 'signal'] = 'weak_buy'
    df.loc[has_pattern & is_uptrend & is_bullish, 'signal_strength'] = 'BUY 🟢'
    df.loc[has_pattern & is_downtrend & is_bearish, 'signal'] = 'weak_sell'
    df.loc[has_pattern & is_downtrend & is_bearish, 'signal_strength'] = 'SELL 🔴'
    
    return df

def get_phase_zones_fast(df: pd.DataFrame) -> list:
    """Optimized zone detection using numpy."""
    if df.empty or 'phase' not in df.columns:
        return []
    
    zones = []
    phase_mask = df['phase'].isin(['accumulation', 'distribution'])
    
    if not phase_mask.any():
        return zones
    
    # Find zone boundaries using diff
    df_filtered = df[phase_mask].copy()
    if df_filtered.empty:
        return zones
    
    # Group consecutive same phases
    df_filtered['group'] = (df_filtered['phase'] != df_filtered['phase'].shift()).cumsum()
    
    for _, group in df_filtered.groupby('group'):
        zones.append({
            'phase': group['phase'].iloc[0],
            'start': group['timestamp'].iloc[0],
            'end': group['timestamp'].iloc[-1],
        })
    
    return zones  # Return all zones

def find_closest_price(df: pd.DataFrame, target_date: pd.Timestamp, current_price: float) -> tuple:
    """Find price closest to target date, return (price, actual_days_diff)."""
    if df.empty:
        return current_price, 0
    
    # Calculate absolute time difference
    df_copy = df.copy()
    df_copy['time_diff'] = abs(df_copy['timestamp'] - target_date)
    
    # Find row with minimum time difference
    closest_idx = df_copy['time_diff'].idxmin()
    closest_price = df_copy.loc[closest_idx, 'close']
    actual_date = df_copy.loc[closest_idx, 'timestamp']
    
    # Calculate actual days difference for validation
    actual_days = abs((actual_date - target_date).days)
    
    return closest_price, actual_days

def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate multi-period performance returns using closest timestamp matching."""
    if df.empty or len(df) < 2:
        return {'7d': 0, '30d': 0, '90d': 0, 'ytd': 0, 'data_age_days': 999, 
                'warnings': ['Insufficient data']}
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    current_price = df['close'].iloc[-1]
    current_date = pd.to_datetime(df['timestamp'].iloc[-1])
    
    # Calculate data age
    try:
        now = pd.Timestamp.now(tz=None)
        data_date = current_date.tz_localize(None) if current_date.tzinfo else current_date
        data_age_days = (now - data_date).days
    except:
        data_age_days = 0
    
    warnings = []
    if data_age_days > 1:
        warnings.append(f'Data is {data_age_days} days old')
    
    # 7D return with closest matching
    target_7d = current_date - pd.Timedelta(days=7)
    price_7d, days_diff_7d = find_closest_price(df, target_7d, current_price)
    if days_diff_7d > 2:  # More than 2 days off target
        warnings.append(f'7D return based on {7 + days_diff_7d}D ago (data gaps)')
    ret_7d = safe_pct_change(current_price, price_7d)
    
    # 30D return
    target_30d = current_date - pd.Timedelta(days=30)
    price_30d, days_diff_30d = find_closest_price(df, target_30d, current_price)
    if days_diff_30d > 5:
        warnings.append(f'30D return based on {30 + days_diff_30d}D ago (data gaps)')
    ret_30d = safe_pct_change(current_price, price_30d)
    
    # 90D return
    target_90d = current_date - pd.Timedelta(days=90)
    price_90d, days_diff_90d = find_closest_price(df, target_90d, current_price)
    if days_diff_90d > 10:
        warnings.append(f'90D return based on {90 + days_diff_90d}D ago (data gaps)')
    ret_90d = safe_pct_change(current_price, price_90d)
    
    # YTD return
    current_year = current_date.year
    ytd_data = df[df['timestamp'].dt.year == current_year]
    if len(ytd_data) > 1:
        price_ytd_start = ytd_data['close'].iloc[0]
        ret_ytd = safe_pct_change(current_price, price_ytd_start)
    else:
        ret_ytd = 0
        warnings.append('Insufficient data for YTD calculation')
    
    return {
        '7d': ret_7d, 
        '30d': ret_30d, 
        '90d': ret_90d, 
        'ytd': ret_ytd, 
        'data_age_days': data_age_days,
        'warnings': warnings
    }

def calculate_signal_strength(phase: str, wyckoff: dict, df: pd.DataFrame, lookback: int = 52) -> tuple:
    """
    Calculate 0-100 signal strength score.
    Returns (score, breakdown_dict)
    """
    score = 50  # Start neutral
    breakdown = {}
    
    # Phase score (+/-25)
    phase_scores = {
        'accumulation': 25, 'uptrend': 15, 'neutral': 0, 'downtrend': -15, 'distribution': -25
    }
    phase_score = phase_scores.get(phase, 0)
    score += phase_score
    breakdown['phase'] = phase_score
    
    # Wyckoff score (+/-20)
    wyckoff_scores = {
        'SMART MONEY BUYING': 20, 'TRENDING UP': 15, 'SIDEWAYS': 0,
        'TRENDING DOWN': -15, 'SMART MONEY SELLING': -20
    }
    wyckoff_score = wyckoff_scores.get(wyckoff.get('label', ''), 0)
    score += wyckoff_score
    breakdown['wyckoff'] = wyckoff_score
    
    # A/D momentum score (+/-15)
    if 'ad' in df.columns and len(df) >= lookback:
        recent = df.tail(lookback)
        ad_change = recent['ad'].iloc[-1] - recent['ad'].iloc[0]
        price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
        
        if price_change < 0 and ad_change > 0:  # Bullish divergence
            ad_score = 15
        elif price_change > 0 and ad_change < 0:  # Bearish divergence
            ad_score = -15
        elif ad_change > 0:
            ad_score = 10
        elif ad_change < 0:
            ad_score = -10
        else:
            ad_score = 0
        score += ad_score
        breakdown['ad_momentum'] = ad_score
    
    # Trend strength bonus (+/-10) using simple momentum
    if len(df) >= 20:
        sma_20 = df['close'].tail(20).mean()
        current = df['close'].iloc[-1]
        if current > sma_20 * 1.05:  # 5% above SMA
            trend_score = 10
        elif current < sma_20 * 0.95:  # 5% below SMA
            trend_score = -10
        else:
            trend_score = 0
        score += trend_score
        breakdown['trend'] = trend_score
    
    # Clamp to 0-100
    score = max(0, min(100, score))
    
    return score, breakdown

def find_support_resistance(df: pd.DataFrame, lookback: int = 52) -> dict:
    """Find key support and resistance levels."""
    if len(df) < lookback:
        return {'support': 0, 'resistance': 0, 'entry_low': 0, 'entry_high': 0}
    
    recent = df.tail(lookback)
    current_price = recent['close'].iloc[-1]
    high = recent['high'].max()
    low = recent['low'].min()
    
    # Simple pivot points
    pivot = (high + low + current_price) / 3
    support1 = 2 * pivot - high
    resistance1 = 2 * pivot - low
    
    # Ensure support < resistance (can be inverted in volatile markets)
    if support1 > resistance1:
        support1, resistance1 = resistance1, support1
    
    # Entry zone (around support)
    entry_low = max(0, support1)  # Ensure non-negative
    entry_high = support1 + (pivot - support1) * 0.3
    
    return {
        'support': support1,
        'resistance': resistance1,
        'entry_low': entry_low,
        'entry_high': entry_high,
        'pivot': pivot
    }

def fetch_symbol_status_enhanced(symbol: str, interval: str, lookback: int, data_source: str = 'yahoo') -> dict:
    """Fetch comprehensive investment data for a single symbol."""
    try:
        df = fetch_data_raw(symbol, interval, data_source)
        if df.empty:
            return {
                'symbol': symbol, 'status': '❓', 'phase': 'No Data', 
                'price': 0, 'change': 0, 'wyckoff': 'N/A', 'wyckoff_emoji': '❓',
                '7d': 0, '30d': 0, '90d': 0, 'ytd': 0,
                'signal_score': 0, 'levels': {}, 'action': 'NO DATA'
            }
        
        phase, _, df = analyze_ad_phase_fast(df, lookback)
        wyckoff = detect_wyckoff_enhanced(df, lookback)
        
        last_price = df.iloc[-1]['close']
        prev_price = df.iloc[-2]['close'] if len(df) > 1 else last_price
        pct_change = safe_pct_change(last_price, prev_price)
        
        # Performance metrics
        perf = calculate_performance_metrics(df)
        
        # Signal strength
        signal_score, score_breakdown = calculate_signal_strength(phase, wyckoff, df, lookback)
        
        # Support/Resistance
        levels = find_support_resistance(df, lookback)
        
        # Action recommendation
        if signal_score >= 80:
            action = '🟢 STRONG BUY'
        elif signal_score >= 60:
            action = '🟡 BUY'
        elif signal_score >= 40:
            action = '⚪ HOLD'
        elif signal_score >= 20:
            action = '🟠 CAUTION'
        else:
            action = '🔴 AVOID'
        
        status_map = {'accumulation': '🟢', 'distribution': '🔴', 'uptrend': '📈', 'downtrend': '📉', 'neutral': '⚪'}
        
        return {
            'symbol': symbol,
            'status': status_map.get(phase, '⚪'),
            'phase': phase.title(),
            'wyckoff': wyckoff['label'],
            'wyckoff_emoji': wyckoff['emoji'],
            'wyckoff_desc': wyckoff['description'],
            'price': last_price,
            'change': pct_change,
            '7d': perf['7d'],
            '30d': perf['30d'],
            '90d': perf['90d'],
            'ytd': perf['ytd'],
            'signal_score': signal_score,
            'score_breakdown': score_breakdown,
            'levels': levels,
            'action': action
        }
    except Exception as e:
        return {
            'symbol': symbol, 'status': '❌', 'phase': 'Error', 
            'price': 0, 'change': 0, 'wyckoff': 'N/A', 'wyckoff_emoji': '❌',
            '7d': 0, '30d': 0, '90d': 0, 'ytd': 0,
            'signal_score': 0, 'levels': {}, 'action': 'ERROR'
        }

def get_watchlist_status_parallel(symbols: list, interval: str = '1wk', lookback: int = 52, data_source: str = 'yahoo') -> list:
    """Fetch watchlist status SEQUENTIALLY - yfinance has thread-safety issues with ThreadPoolExecutor."""
    results = []
    for sym in symbols:
        result = fetch_symbol_status_enhanced(sym, interval, lookback, data_source)
        results.append(result)
    return results

def calculate_mtf_alignment(df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> dict:
    """Calculate multi-timeframe alignment score (0-100)."""
    score = 50
    factors = {}
    
    if df_daily.empty or df_weekly.empty:
        return {'score': 0, 'factors': {}, 'recommendation': 'INSUFFICIENT DATA', 'confidence': 'LOW'}
    
    # Daily analysis
    phase_d, _, df_daily = analyze_ad_phase_fast(df_daily, 26)
    phase_w, _, df_weekly = analyze_ad_phase_fast(df_weekly, 52)
    
    # Phase alignment (40 points)
    bullish_phases = ['accumulation', 'uptrend']
    bearish_phases = ['distribution', 'downtrend']
    
    if phase_d in bullish_phases and phase_w in bullish_phases:
        phase_align = 40
    elif phase_d in bearish_phases and phase_w in bearish_phases:
        phase_align = -40
    elif (phase_d in bullish_phases) != (phase_w in bullish_phases):
        phase_align = 0  # Mixed - neutral
    else:
        phase_align = 0
    
    score += phase_align // 2
    factors['phase_alignment'] = phase_align
    
    # Trend direction (30 points) - check length BEFORE accessing index
    if len(df_daily) >= 20:
        daily_trend = 1 if df_daily['close'].iloc[-1] > df_daily['close'].iloc[-20] else -1
    else:
        daily_trend = 0
    
    if len(df_weekly) >= 10:
        weekly_trend = 1 if df_weekly['close'].iloc[-1] > df_weekly['close'].iloc[-10] else -1
    else:
        weekly_trend = 0
    
    if daily_trend == weekly_trend == 1:
        trend_align = 30
    elif daily_trend == weekly_trend == -1:
        trend_align = -30
    else:
        trend_align = 0
    
    score += trend_align // 2
    factors['trend_alignment'] = trend_align
    
    # A/D momentum (30 points) - with proper bounds checking using safe_series_diff
    if 'ad' in df_daily.columns and 'ad' in df_weekly.columns:
        ad_daily = safe_series_diff(df_daily['ad'], 20)
        ad_weekly = safe_series_diff(df_weekly['ad'], 10)
        
        if ad_daily > 0 and ad_weekly > 0:
            ad_align = 30
        elif ad_daily < 0 and ad_weekly < 0:
            ad_align = -30
        else:
            ad_align = 0
        
        score += ad_align // 2
        factors['ad_alignment'] = ad_align
    
    # Clamp score
    score = max(0, min(100, score))
    
    # Recommendation
    if score >= 75:
        rec = 'STRONG BUY ZONE'
        conf = 'HIGH'
    elif score >= 60:
        rec = 'FAVORABLE ENTRY'
        conf = 'MEDIUM-HIGH'
    elif score >= 40:
        rec = 'NEUTRAL - WAIT'
        conf = 'MEDIUM'
    elif score >= 25:
        rec = 'UNFAVORABLE'
        conf = 'MEDIUM'
    else:
        rec = 'STRONG SELL ZONE'
        conf = 'HIGH'
    
    return {'score': score, 'factors': factors, 'recommendation': rec, 'confidence': conf}

def calculate_trend_strength_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """Calculate ADX-based trend strength."""
    if len(df) < period * 2:
        return {'adx': 0, 'strength': 'INSUFFICIENT DATA', 'direction': 'neutral'}
    
    try:
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        plus_di = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        minus_di = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
        current_plus = plus_di[-1] if not np.isnan(plus_di[-1]) else 0
        current_minus = minus_di[-1] if not np.isnan(minus_di[-1]) else 0
        
        # Trend strength
        if current_adx >= 50:
            strength = '🔥 VERY STRONG'
        elif current_adx >= 25:
            strength = '📈 STRONG'
        elif current_adx >= 20:
            strength = '↗️ MODERATE'
        else:
            strength = '↔️ WEAK/RANGING'
        
        # Direction
        if current_plus > current_minus:
            direction = 'bullish'
        elif current_minus > current_plus:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {'adx': current_adx, 'strength': strength, 'direction': direction, 'plus_di': current_plus, 'minus_di': current_minus}
    except:
        return {'adx': 0, 'strength': 'ERROR', 'direction': 'neutral'}

# --- Validation and Warning Functions ---
def validate_symbol(symbol: str) -> tuple:
    """
    Validate symbol format.
    Returns: (is_valid: bool, cleaned_symbol: str, message: str)
    """
    if not symbol:
        return False, "", "Symbol cannot be empty"
    
    # Remove whitespace
    symbol = symbol.strip().upper()
    
    # Check length
    if len(symbol) < 3 or len(symbol) > 20:
        return False, symbol, "Symbol must be 3-20 characters"
    
    # Check format
    if '-' in symbol:
        parts = symbol.split('-')
        if len(parts) != 2:
            return False, symbol, "Use format: BTC-USD"
        if not parts[0].isalpha() or not parts[1].isalpha():
            return False, symbol, "Symbol parts must be letters only"
    else:
        if not symbol.replace('USDT', '').replace('USD', '').isalpha():
            return False, symbol, "Invalid symbol format"
    
    return True, symbol, ""

def show_data_freshness_warning(data_age_days: int, warnings: list = None):
    """Display prominent warning if data is stale."""
    if data_age_days > 2:
        st.error(f"""
        ⚠️ **DATA FRESHNESS ALERT**
        
        Latest data is **{data_age_days} days old**
        
        Investment signals may not reflect current market conditions.
        """)
    elif data_age_days > 0:
        st.warning(f"ℹ️ Data is {data_age_days} day(s) old")
    
    if warnings:
        with st.expander("⚠️ Data Quality Warnings"):
            for warning in warnings:
                st.warning(warning)

# --- UI ---
st.title("🕯️ Crypto Pattern Watcher")

# PROMINENT DISCLAIMER
st.error("""
⚠️ **INVESTMENT DISCLAIMER - READ CAREFULLY**

This tool is for **EDUCATIONAL PURPOSES ONLY**. It is NOT financial advice.

**Important:**
- All signals are based on technical indicators that have NOT been validated for profitability
- Past performance does NOT guarantee future results
- Cryptocurrency is EXTREMELY volatile and risky
- You can lose ALL your invested capital
- Only invest what you can afford to lose completely
- Consult a licensed financial advisor before making investment decisions

**The developers assume NO responsibility for financial losses.**

By using this tool, you acknowledge these risks.
""")

st.caption("Long-term A/D Analysis | Multi-Asset Watchlist | Entry Signals | Simplified Wyckoff")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Data Source Toggle
    st.subheader("📡 Data Source")
    data_source_options = ['yahoo', 'cryptocompare']
    data_source_labels = {'yahoo': 'Yahoo Finance', 'cryptocompare': 'CryptoCompare'}
    
    # Create toggle with Yahoo Finance on left (index 0) as default
    col_left, col_right = st.columns(2)
    with col_left:
        yahoo_selected = st.session_state.data_source == 'yahoo'
        if st.button(
            "📊 Yahoo Finance" if yahoo_selected else "Yahoo Finance",
            use_container_width=True,
            type="primary" if yahoo_selected else "secondary"
        ):
            st.session_state.data_source = 'yahoo'
            st.rerun()
    with col_right:
        crypto_selected = st.session_state.data_source == 'cryptocompare'
        if st.button(
            "🔷 CryptoCompare" if crypto_selected else "CryptoCompare",
            use_container_width=True,
            type="primary" if crypto_selected else "secondary"
        ):
            st.session_state.data_source = 'cryptocompare'
            st.rerun()
    
    st.caption(f"Current: **{data_source_labels[st.session_state.data_source]}**")
    
    # Manual cache clearing on data source change
    if 'previous_data_source' not in st.session_state:
        st.session_state.previous_data_source = st.session_state.data_source
    
    if st.session_state.data_source != st.session_state.previous_data_source:
        st.cache_data.clear()
        st.session_state.previous_data_source = st.session_state.data_source
        st.info("🔄 Cache cleared due to data source change")
    
    # Data source recommendation note
    if st.session_state.data_source == 'cryptocompare':
        st.caption("⚠️ *For Weekly timeframe, Yahoo Finance is recommended (native weekly data)*")
    
    st.divider()
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ["📊 Single Asset", "📋 Watchlist Dashboard", "🔄 Timeframe Compare", "📈 Open Interest Monitor"],
        index=0
    )
    
    st.divider()
    
    if analysis_mode == "📊 Single Asset":
        symbol_input = st.text_input("Symbol", value="BTC-USD")
        is_valid, symbol, msg = validate_symbol(symbol_input)
        if not is_valid and symbol_input:
            st.error(f"❌ {msg}")
        symbol = symbol if is_valid else symbol_input.upper()
        
        interval = st.selectbox("Timeframe", ["1d", "1wk"], format_func=lambda x: "Daily" if x == "1d" else "Weekly")
        
        lookback_presets = {"Short (10)": 10, "Mid (26)": 26, "Long (52)": 52}
        lookback_selection = st.selectbox("Lookback", options=list(lookback_presets.keys()), index=2)
        lookback_period = lookback_presets[lookback_selection]
        
        analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary", disabled=not is_valid)
    
    elif analysis_mode == "📋 Watchlist Dashboard":
        st.subheader("Manage Watchlist")
        new_symbol = st.text_input("Add Symbol", placeholder="e.g., AVAX-USD")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add", use_container_width=True):
                if new_symbol and new_symbol.upper() not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_symbol.upper())
                    st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.watchlist = []
                st.rerun()
        
        st.caption(f"Watchlist ({len(st.session_state.watchlist)}):")
        for i, sym in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([3, 1])
            col1.write(sym)
            if col2.button("❌", key=f"del_{i}"):
                st.session_state.watchlist.remove(sym)
                st.rerun()
        
        refresh_btn = st.button("🔄 Refresh", use_container_width=True, type="primary")
    
    elif analysis_mode == "🔄 Timeframe Compare":
        symbol = st.text_input("Symbol", value="BTC-USD").upper()
        compare_btn = st.button("🔄 Compare", use_container_width=True, type="primary")
    
    elif analysis_mode == "📈 Open Interest Monitor":
        symbol = st.text_input("Symbol", value="BTC-USD", key="oi_symbol").upper()
        
        # Coinalyze intervals: 1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 12hour, daily
        oi_period = st.selectbox(
            "OI Resolution",
            ["4hour", "daily"],
            format_func=lambda x: {"4hour": "4-Hour (~3 months)", "daily": "Daily (1 year)"}[x],
            index=1
        )
        
        # Set max days based on period selection
        oi_days = 90 if oi_period == "4hour" else 365
        
        st.divider()
        st.subheader("🔑 API Status")
        
        api_configured = bool(os.environ.get('COINALYZE_API_KEY', ''))
        if api_configured:
            st.success("✅ Coinalyze API configured")
        else:
            st.error("❌ COINALYZE_API_KEY not set")
        
        oi_analyze_btn = st.button("📈 Analyze OI", use_container_width=True, type="primary")

# --- Main Content ---

# Single Asset
if analysis_mode == "📊 Single Asset" and 'analyze_btn' in dir() and analyze_btn:
    try:
        with st.spinner(f"Analyzing {symbol}..."):
            df = fetch_data(symbol, interval, st.session_state.data_source)
        
        if df.empty:
            st.error(f"""
            ❌ No data available for {symbol}
            
            Possible reasons:
            - Symbol not found
            - Data source issues
            - Network connectivity
            
            Try:
            - Checking symbol format (BTC-USD)
            - Switching data sources
            - Trying a different symbol
            """)
            st.stop()
        
        # Validation
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Data missing required columns: {missing_cols}")
            st.stop()
        
        if len(df) < lookback_period:
            st.warning(f"""
            ⚠️ Insufficient data for {lookback_period} period analysis
            
            Available: {len(df)} periods
            Required: {lookback_period} periods
            
            Results may be less reliable.
            """)
        
        # Calculate performance metrics first to check data freshness
        perf = calculate_performance_metrics(df)
        show_data_freshness_warning(perf['data_age_days'], perf.get('warnings'))
        
        df = detect_patterns_optimized(df)
        phase, color, df = analyze_ad_phase_fast(df, lookback=lookback_period)
        df = generate_signals_fast(df)
        wyckoff = detect_wyckoff_enhanced(df, lookback_period)
        zones = get_phase_zones_fast(df)
        
        # Metrics
        last = df.iloc[-1]
        prev = df.iloc[-2]
        pct_change = ((last['close'] - prev['close']) / prev['close']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{symbol}", f"${last['close']:,.2f}", f"{pct_change:+.2f}%")
        col2.metric("Phase", phase.title())
        col3.metric("Wyckoff", wyckoff['phase'])
        col4.metric("Outlook", wyckoff['label'])
        
        # Wyckoff Card
        with st.expander(f"{wyckoff['emoji']} {wyckoff['phase']} - What This Means", expanded=True):
            st.markdown(f"### {wyckoff['label']}\n\n{wyckoff['description']}")
            if wyckoff['color'] == 'green':
                st.success("✅ Favorable for accumulating positions")
            elif wyckoff['color'] == 'red':
                st.error("⚠️ Consider taking profits or waiting")
            else:
                st.info("⏳ Wait for clearer direction")
        
        # Price Chart with Zones
        st.subheader("📈 Price Chart with A/D Zones")
        fig = go.Figure()
        
        for zone in zones:
            zone_color = 'rgba(0, 255, 0, 0.1)' if zone['phase'] == 'accumulation' else 'rgba(255, 0, 0, 0.1)'
            fig.add_vrect(x0=zone['start'], x1=zone['end'], fillcolor=zone_color, layer="below", line_width=0)
        
        # Full chart data - no limits
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], 
            low=df['low'], close=df['close'], name=symbol,
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))
        
        # Entry signals - full data
        strong_buys = df[df['signal'] == 'strong_buy']
        strong_sells = df[df['signal'] == 'strong_sell']
        
        if not strong_buys.empty:
            fig.add_trace(go.Scatter(
                x=strong_buys['timestamp'], y=strong_buys['low'] * 0.97,
                mode='markers', name='Strong Buy',
                marker=dict(color='lime', size=12, symbol='star'),
                hoverinfo='skip'  # Don't block hover on candlesticks
            ))
        
        if not strong_sells.empty:
            fig.add_trace(go.Scatter(
                x=strong_sells['timestamp'], y=strong_sells['high'] * 1.03,
                mode='markers', name='Strong Sell',
                marker=dict(color='red', size=12, symbol='star'),
                hoverinfo='skip'  # Don't block hover on candlesticks
            ))
        
        fig.update_layout(height=450, xaxis_rangeslider_visible=False, template='plotly_dark', dragmode='pan')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        # A/D Chart
        st.subheader("💰 Money Flow (A/D Line)")
        
        ad_recent = df.tail(lookback_period)
        ad_trend = ad_recent['ad'].iloc[-1] - ad_recent['ad'].iloc[0]
        ad_trend_pct = (ad_trend / abs(ad_recent['ad'].iloc[0])) * 100 if ad_recent['ad'].iloc[0] != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend", f"{'📈 IN' if ad_trend > 0 else '📉 OUT'}")
        with col2:
            st.metric("A/D Change", f"{ad_trend_pct:+.1f}%")
        with col3:
            price_chg = df['close'].iloc[-1] - df['close'].iloc[-lookback_period] if len(df) > lookback_period else 0
            if price_chg < 0 and ad_trend > 0:
                st.success("🔍 Bullish Divergence")
            elif price_chg > 0 and ad_trend < 0:
                st.error("🔍 Bearish Divergence")
            else:
                st.info("🔍 Aligned")
        
        # A/D chart - full data
        ad_fig = go.Figure()
        ad_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ad'], name='A/D Line', 
                                    line=dict(color='orange', width=2), fill='tozeroy', fillcolor='rgba(255,165,0,0.1)'))
        ad_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ad_ema'], name='EMA 21', 
                                    line=dict(color='yellow', dash='dot', width=1)))
        ad_fig.update_layout(height=250, template='plotly_dark')
        st.plotly_chart(ad_fig, use_container_width=True, config={'scrollZoom': True})
        
        with st.expander("📖 How to Read"):
            st.markdown("""
**A/D Line Rising** = Money flowing IN (bullish) | **Falling** = OUT (bearish)

**Divergences (key signals):**
- Price ↓ but A/D ↑ = 🟢 Bullish - accumulate
- Price ↑ but A/D ↓ = 🔴 Bearish - take profits
            """)
        
        # Signals Table
        st.subheader("📊 Recent Signals")
        signals = df[df['signal'] != 'none'].tail(10).sort_values('timestamp', ascending=False)
        if not signals.empty:
            st.dataframe(signals[['timestamp', 'close', 'candlestick_pattern', 'phase', 'signal_strength']].rename(
                columns={'timestamp': 'Date', 'close': 'Price', 'candlestick_pattern': 'Pattern', 'phase': 'Phase', 'signal_strength': 'Signal'}
            ), use_container_width=True, hide_index=True)
        else:
            st.info("No signals in visible range")
    
    except Exception as e:
        st.error(f"""
        ❌ Analysis failed
        
        Error: {str(e)}
        
        Please try:
        - Refreshing the page
        - Selecting a different symbol
        - Changing data source
        - Reporting this issue if it persists
        """)
        with st.expander("🔍 Technical Details"):
            st.exception(e)

# Watchlist
elif analysis_mode == "📋 Watchlist Dashboard":
    st.subheader("📋 Investment Watchlist Dashboard")
    
    if not st.session_state.watchlist:
        st.info("Watchlist empty. Add symbols via sidebar.")
    elif 'refresh_btn' in dir() and refresh_btn:
        with st.spinner("Fetching comprehensive data..."):
            data = get_watchlist_status_parallel(st.session_state.watchlist, '1wk', 52, st.session_state.data_source)
        
        # --- OPPORTUNITY HIGHLIGHT CARDS ---
        st.markdown("### 🎯 Quick Insights")
        
        # Find best opportunity (highest score in accumulation)
        accum_assets = [d for d in data if d['phase'] == 'Accumulation']
        best_opp = max(accum_assets, key=lambda x: x['signal_score']) if accum_assets else None
        
        # Find distribution alerts
        distrib_assets = [d for d in data if d['phase'] == 'Distribution']
        
        # Find top performer (best 30D return)
        valid_data = [d for d in data if d['price'] > 0]
        top_performer = max(valid_data, key=lambda x: x['30d']) if valid_data else None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if best_opp:
                st.success(f"""
                🏆 **Best Opportunity**  
                **{best_opp['symbol']}**  
                Score: {best_opp['signal_score']}/100  
                {best_opp['action']}
                """)
            else:
                st.info("🏆 No accumulation opportunities found")
        
        with col2:
            if distrib_assets:
                symbols = ", ".join([d['symbol'] for d in distrib_assets[:3]])
                st.error(f"""
                ⚠️ **Distribution Alert**  
                {len(distrib_assets)} asset(s) in distribution  
                {symbols}
                """)
            else:
                st.success("✅ No distribution alerts")
        
        with col3:
            if top_performer and top_performer['30d'] != 0:
                color = "green" if top_performer['30d'] > 0 else "red"
                st.info(f"""
                📈 **Top 30D Performer**  
                **{top_performer['symbol']}**  
                {top_performer['30d']:+.1f}%
                """)
            else:
                st.info("📈 Performance data loading...")
        
        st.divider()
        
        # --- SUMMARY METRICS ---
        accum = sum(1 for w in data if w['phase'] == 'Accumulation')
        distrib = sum(1 for w in data if w['phase'] == 'Distribution')
        avg_score = sum(d['signal_score'] for d in data) / len(data) if data else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Accumulation", accum)
        col2.metric("🔴 Distribution", distrib)
        col3.metric("📊 Avg Signal", f"{avg_score:.0f}/100")
        col4.metric("📋 Total Assets", len(data))
        
        st.divider()
        
        # --- SORTABLE DATA TABLE ---
        st.markdown("### 📊 Performance Overview")
        
        # Build table data
        table_data = []
        for item in data:
            levels = item.get('levels', {})
            support = levels.get('support', 0)
            resistance = levels.get('resistance', 0)
            
            table_data.append({
                'Asset': f"{item['status']} {item['symbol']}",
                'Price': f"${item['price']:,.2f}" if item['price'] > 0 else "N/A",
                '7D': item['7d'],
                '30D': item['30d'],
                '90D': item['90d'],
                'YTD': item['ytd'],
                'Phase': item['phase'],
                'Score': item['signal_score'],
                'Action': item['action'],
                'Support': f"${support:,.0f}" if support > 0 else "-",
                'Resistance': f"${resistance:,.0f}" if resistance > 0 else "-"
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Format percentage columns with colors
        def color_returns(val):
            if isinstance(val, (int, float)):
                color = 'color: #26a69a' if val > 0 else 'color: #ef5350' if val < 0 else ''
                return color
            return ''
        
        def color_score(val):
            if val >= 80:
                return 'background-color: rgba(38, 166, 154, 0.3)'
            elif val >= 60:
                return 'background-color: rgba(255, 235, 59, 0.3)'
            elif val >= 40:
                return ''
            elif val >= 20:
                return 'background-color: rgba(255, 152, 0, 0.3)'
            else:
                return 'background-color: rgba(239, 83, 80, 0.3)'
        
        # Display with column config
        st.dataframe(
            df_table,
            column_config={
                'Asset': st.column_config.TextColumn('Asset', width='medium'),
                'Price': st.column_config.TextColumn('Price', width='small'),
                '7D': st.column_config.NumberColumn('7D %', format="%.1f%%"),
                '30D': st.column_config.NumberColumn('30D %', format="%.1f%%"),
                '90D': st.column_config.NumberColumn('90D %', format="%.1f%%"),
                'YTD': st.column_config.NumberColumn('YTD %', format="%.1f%%"),
                'Phase': st.column_config.TextColumn('Phase', width='small'),
                'Score': st.column_config.ProgressColumn('Signal', min_value=0, max_value=100, format="%d"),
                'Action': st.column_config.TextColumn('Action', width='medium'),
                'Support': st.column_config.TextColumn('Support', width='small'),
                'Resistance': st.column_config.TextColumn('Resistance', width='small'),
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # --- DETAILED CARDS ---
        st.markdown("### 📋 Detailed Analysis")
        
        # Sort by signal score descending
        sorted_data = sorted(data, key=lambda x: x['signal_score'], reverse=True)
        
        for item in sorted_data:
            with st.expander(f"{item['status']} **{item['symbol']}** - Score: {item['signal_score']}/100 | {item['action']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📈 Performance**")
                    metrics = f"""
                    - 7D: {item['7d']:+.1f}%
                    - 30D: {item['30d']:+.1f}%
                    - 90D: {item['90d']:+.1f}%
                    - YTD: {item['ytd']:+.1f}%
                    """
                    st.markdown(metrics)
                
                with col2:
                    st.markdown("**🔍 Analysis**")
                    st.markdown(f"""
                    - Phase: {item['phase']}
                    - Wyckoff: {item['wyckoff_emoji']} {item['wyckoff']}
                    - {item.get('wyckoff_desc', '')}
                    """)
                
                with col3:
                    levels = item.get('levels', {})
                    if levels:
                        st.markdown("**🎯 Key Levels**")
                        st.markdown(f"""
                        - Support: ${levels.get('support', 0):,.2f}
                        - Resistance: ${levels.get('resistance', 0):,.2f}
                        - Entry Zone: ${levels.get('entry_low', 0):,.2f} - ${levels.get('entry_high', 0):,.2f}
                        """)
        
        # --- LEGEND ---
        with st.expander("📖 Signal Score Guide"):
            st.markdown("""
            | Score | Action | Meaning |
            |-------|--------|---------|
            | 80-100 | 🟢 STRONG BUY | Accumulation + Bullish signals aligned |
            | 60-79 | 🟡 BUY | Favorable conditions |
            | 40-59 | ⚪ HOLD | Neutral - wait for clarity |
            | 20-39 | 🟠 CAUTION | Unfavorable signals |
            | 0-19 | 🔴 AVOID | Distribution + Bearish signals |
            
            **Factors considered:** Phase, Wyckoff, A/D Momentum, Trend Position
            """)
    else:
        st.info("👆 Click 'Refresh' to load data")

# Timeframe Compare
elif analysis_mode == "🔄 Timeframe Compare":
    st.subheader(f"🔄 {symbol} - Multi-Timeframe Analysis")
    
    if 'compare_btn' in dir() and compare_btn:
        with st.spinner("Analyzing daily and weekly timeframes..."):
            df_d = fetch_data(symbol, '1d', st.session_state.data_source)
            df_w = fetch_data(symbol, '1wk', st.session_state.data_source)
        
        if not df_d.empty and not df_w.empty:
            # Calculate alignment score
            alignment = calculate_mtf_alignment(df_d.copy(), df_w.copy())
            
            # Re-analyze for display (since mtf_alignment modifies dfs)
            phase_d, _, df_d = analyze_ad_phase_fast(df_d, 26)
            phase_w, _, df_w = analyze_ad_phase_fast(df_w, 52)
            wyck_d = detect_wyckoff_enhanced(df_d, 26)
            wyck_w = detect_wyckoff_enhanced(df_w, 52)
            
            # Trend strength
            trend_d = calculate_trend_strength_adx(df_d)
            trend_w = calculate_trend_strength_adx(df_w)
            
            # Key levels (from weekly)
            levels = find_support_resistance(df_w, 52)
            current_price = df_w['close'].iloc[-1]
            
            # --- ALIGNMENT SCORE BANNER ---
            score = alignment['score']
            rec = alignment['recommendation']
            conf = alignment['confidence']
            
            if score >= 75:
                st.success(f"""
                ### 🎯 ALIGNMENT SCORE: {score}/100 - {rec}
                **Confidence: {conf}** | Both timeframes aligned bullish - favorable for entries
                """)
            elif score >= 60:
                st.info(f"""
                ### 🎯 ALIGNMENT SCORE: {score}/100 - {rec}
                **Confidence: {conf}** | Conditions improving - monitor for entry
                """)
            elif score >= 40:
                st.warning(f"""
                ### 🎯 ALIGNMENT SCORE: {score}/100 - {rec}
                **Confidence: {conf}** | Mixed signals - wait for clarity
                """)
            else:
                st.error(f"""
                ### 🎯 ALIGNMENT SCORE: {score}/100 - {rec}
                **Confidence: {conf}** | Unfavorable conditions - avoid or reduce exposure
                """)
            
            st.divider()
            
            # --- TIMEFRAME COMPARISON CARDS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Daily Timeframe")
                
                # Phase & Wyckoff
                phase_color = "🟢" if phase_d in ['accumulation', 'uptrend'] else "🔴" if phase_d in ['distribution', 'downtrend'] else "⚪"
                st.metric("Phase", f"{phase_color} {phase_d.title()}")
                
                # Trend strength
                trend_dir_emoji = "📈" if trend_d['direction'] == 'bullish' else "📉" if trend_d['direction'] == 'bearish' else "↔️"
                st.metric("Trend", f"{trend_d['strength']}", delta=f"ADX: {trend_d['adx']:.1f}")
                
                # A/D status
                if 'ad' in df_d.columns and len(df_d) >= 20:
                    ad_recent = df_d['ad'].iloc[-1] - df_d['ad'].iloc[-20]
                    ad_status = "📈 Money Flowing IN" if ad_recent > 0 else "📉 Money Flowing OUT"
                    st.caption(ad_status)
                
                st.caption(wyck_d['description'])
            
            with col2:
                st.markdown("### 📆 Weekly Timeframe")
                
                # Phase & Wyckoff
                phase_color = "🟢" if phase_w in ['accumulation', 'uptrend'] else "🔴" if phase_w in ['distribution', 'downtrend'] else "⚪"
                st.metric("Phase", f"{phase_color} {phase_w.title()}")
                
                # Trend strength
                trend_dir_emoji = "📈" if trend_w['direction'] == 'bullish' else "📉" if trend_w['direction'] == 'bearish' else "↔️"
                st.metric("Trend", f"{trend_w['strength']}", delta=f"ADX: {trend_w['adx']:.1f}")
                
                # A/D status
                if 'ad' in df_w.columns and len(df_w) >= 10:
                    ad_recent = df_w['ad'].iloc[-1] - df_w['ad'].iloc[-10]
                    ad_status = "📈 Money Flowing IN" if ad_recent > 0 else "📉 Money Flowing OUT"
                    st.caption(ad_status)
                
                st.caption(wyck_w['description'])
            
            st.divider()
            
            # --- KEY LEVELS & ENTRY ZONE ---
            st.markdown("### 🎯 Key Levels & Entry Zone")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:,.2f}")
            
            with col2:
                support = levels.get('support', 0)
                distance_to_support = ((current_price - support) / current_price) * 100 if support > 0 else 0
                st.metric("Strong Support", f"${support:,.2f}", delta=f"{distance_to_support:+.1f}% away")
            
            with col3:
                resistance = levels.get('resistance', 0)
                distance_to_resistance = ((resistance - current_price) / current_price) * 100 if resistance > 0 else 0
                st.metric("Resistance", f"${resistance:,.2f}", delta=f"{distance_to_resistance:+.1f}% away")
            
            with col4:
                pivot = levels.get('pivot', 0)
                st.metric("Pivot", f"${pivot:,.2f}")
            
            # Entry zone recommendation
            entry_low = levels.get('entry_low', 0)
            entry_high = levels.get('entry_high', 0)
            
            if score >= 60 and entry_low > 0:
                st.success(f"""
                **🎯 Optimal Entry Zone:** ${entry_low:,.2f} - ${entry_high:,.2f}
                
                *Strategy: Consider DCA (Dollar Cost Averaging) if price enters this zone*
                """)
            elif score >= 40:
                st.info(f"""
                **⏳ Wait Zone:** Market conditions mixed
                
                *Strategy: Wait for clearer signals before entering*
                """)
            else:
                st.warning(f"""
                **⚠️ Caution Zone:** Unfavorable conditions
                
                *Strategy: Avoid new entries, consider reducing exposure if in profit*
                """)
            
            st.divider()
            
            # --- CHARTS ---
            st.markdown("### 📊 Price Charts")
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Daily", "Weekly"), vertical_spacing=0.12)
            
            # Daily chart with key levels
            fig.add_trace(go.Candlestick(
                x=df_d['timestamp'], open=df_d['open'], high=df_d['high'],
                low=df_d['low'], close=df_d['close'], name='Daily',
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            # Weekly chart with support/resistance
            fig.add_trace(go.Candlestick(
                x=df_w['timestamp'], open=df_w['open'], high=df_w['high'],
                low=df_w['low'], close=df_w['close'], name='Weekly',
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=2, col=1)
            
            # Add support/resistance lines to weekly chart
            if levels.get('support', 0) > 0:
                fig.add_hline(y=levels['support'], line_dash="dash", line_color="green", 
                            annotation_text="Support", row=2, col=1)
            if levels.get('resistance', 0) > 0:
                fig.add_hline(y=levels['resistance'], line_dash="dash", line_color="red",
                            annotation_text="Resistance", row=2, col=1)
            
            fig.update_layout(height=700, template='plotly_dark', showlegend=False)
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # --- ALIGNMENT FACTORS BREAKDOWN ---
            with st.expander("📖 Alignment Score Breakdown"):
                factors = alignment.get('factors', {})
                
                st.markdown("""
                | Factor | Score | Description |
                |--------|-------|-------------|
                | Phase Alignment | {} | Both timeframes in same market phase |
                | Trend Alignment | {} | Price direction consistency |
                | A/D Alignment | {} | Money flow consistency |
                """.format(
                    f"+{factors.get('phase_alignment', 0)}" if factors.get('phase_alignment', 0) > 0 else factors.get('phase_alignment', 0),
                    f"+{factors.get('trend_alignment', 0)}" if factors.get('trend_alignment', 0) > 0 else factors.get('trend_alignment', 0),
                    f"+{factors.get('ad_alignment', 0)}" if factors.get('ad_alignment', 0) > 0 else factors.get('ad_alignment', 0)
                ))
                
                st.markdown("""
                **Interpretation:**
                - **75-100:** Strong alignment - high confidence entries
                - **60-74:** Good alignment - favorable conditions
                - **40-59:** Mixed signals - wait for clarity
                - **25-39:** Poor alignment - caution advised
                - **0-24:** Strong bearish alignment - avoid entries
                """)
        else:
            st.error(f"Could not load {symbol}")
    else:
        st.info("👆 Click 'Compare' to analyze")

# Open Interest Monitor
elif analysis_mode == "📈 Open Interest Monitor":
    st.subheader(f"📈 {symbol} - Derivatives Analysis")
    st.caption("Open Interest | Funding Rate | Long/Short Ratio | Investment Score | Powered by Coinalyze")
    
    if 'oi_analyze_btn' in dir() and oi_analyze_btn:
        with st.spinner(f"Fetching derivatives data for {symbol}..."):
            # Fetch all derivatives data from Coinalyze
            oi_current = fetch_open_interest(symbol)
            oi_history = fetch_open_interest_history(symbol, oi_period, oi_days)
            funding = fetch_funding_rate(symbol)
            ls_ratio = fetch_long_short_ratio(symbol)
            df_price = fetch_data(symbol, '1d')
        
        # Debug expander - show what was fetched
        with st.expander("🔧 Debug: API Response Status", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**OI Current:**", "✅" if not oi_current.get('error') else f"❌ {oi_current.get('error')}")
                st.write("**OI History:**", f"✅ {len(oi_history)} rows" if not oi_history.empty else "❌ Empty")
                st.write("**Funding Rate:**", "✅" if not funding.get('error') else f"❌ {funding.get('error')}")
            with col2:
                st.write("**L/S Ratio:**", "✅" if not ls_ratio.get('error') else f"❌ {ls_ratio.get('error')}")
                st.write("**Price Data:**", f"✅ {len(df_price)} rows" if not df_price.empty else "❌ Empty")
                st.write("**API Key:**", "✅ Set" if os.environ.get('COINALYZE_API_KEY') else "❌ Not set")
        
        # Check for critical errors
        has_error = oi_current.get('error') and 'timeout' not in str(oi_current.get('error', '')).lower()
        
        if has_error:
            st.error(f"⚠️ API Error: {oi_current['error']}")
            st.info("""
            **Troubleshooting:**
            - Verify `COINALYZE_API_KEY` is set in Streamlit Secrets
            - Check your API key is valid at coinalyze.net
            - Symbol format: BTC-USD → BTCUSD_PERP.A
            """)
        else:
            # --- CALCULATE ALL METRICS ---
            current_oi = oi_current.get('oi', 0)
            binance_sym = oi_current.get('binance_symbol', symbol)
            
            # OI change
            oi_change_pct = 0
            if not oi_history.empty and len(oi_history) >= 2:
                first_oi = oi_history['sumOpenInterest'].iloc[0]
                last_oi = oi_history['sumOpenInterest'].iloc[-1]
                oi_change_pct = safe_pct_change(last_oi, first_oi)
            
            # Price change (match to OI history length)
            price_change_pct = 0
            current_price = 0
            if not df_price.empty and len(df_price) >= 2:
                lookback = min(len(df_price), len(oi_history)) if not oi_history.empty else 30
                first_price = df_price['close'].iloc[-lookback]
                current_price = df_price['close'].iloc[-1]
                price_change_pct = safe_pct_change(current_price, first_price)
            
            # Funding rate
            funding_rate = funding.get('rate', 0)
            
            # Long/Short ratio
            long_pct = ls_ratio.get('long_pct', 50)
            short_pct = ls_ratio.get('short_pct', 50)
            ratio_value = ls_ratio.get('ratio', 1.0)
            
            # Calculate comprehensive score
            deriv_score = calculate_derivatives_score(oi_change_pct, funding_rate, ratio_value, price_change_pct)
            
            # --- INVESTMENT SCORE BANNER ---
            score = deriv_score['score']
            recommendation = deriv_score['recommendation']
            
            if score >= 70:
                st.success(f"""
                ### 🎯 Investment Score: {score}/100
                **{recommendation}**
                """)
            elif score >= 55:
                st.info(f"""
                ### 🎯 Investment Score: {score}/100
                **{recommendation}**
                """)
            elif score >= 45:
                st.warning(f"""
                ### 🎯 Investment Score: {score}/100
                **{recommendation}**
                """)
            else:
                st.error(f"""
                ### 🎯 Investment Score: {score}/100
                **{recommendation}**
                """)
            
            st.divider()
            
            # --- KEY DERIVATIVES METRICS ---
            st.markdown("### 📊 Derivatives Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Format OI
                if current_oi >= 1_000_000:
                    oi_display = f"{current_oi/1_000_000:.2f}M"
                elif current_oi >= 1_000:
                    oi_display = f"{current_oi/1_000:.2f}K"
                else:
                    oi_display = f"{current_oi:.2f}"
                st.metric("Open Interest", oi_display, delta=f"{oi_change_pct:+.2f}%")
            
            with col2:
                # Funding rate interpretation
                if funding_rate < -0.01:
                    funding_label = f"{funding_rate:.4f}% 🟢"
                elif funding_rate > 0.03:
                    funding_label = f"{funding_rate:.4f}% 🔴"
                else:
                    funding_label = f"{funding_rate:.4f}%"
                st.metric("Funding Rate", funding_label)
            
            with col3:
                # Long/Short interpretation
                if ratio_value > 1.2:
                    ls_label = f"{ratio_value:.2f} (🐂 Longs)"
                elif ratio_value < 0.8:
                    ls_label = f"{ratio_value:.2f} (🐻 Shorts)"
                else:
                    ls_label = f"{ratio_value:.2f} (Balanced)"
                st.metric("Long/Short Ratio", ls_label)
            
            with col4:
                st.metric("Price", f"${current_price:,.2f}" if current_price > 0 else "N/A", delta=f"{price_change_pct:+.1f}%")
            
            st.divider()
            
            # --- SCORE FACTORS BREAKDOWN ---
            st.markdown("### 🔍 Score Breakdown")
            
            factors = deriv_score['factors']
            if factors:
                cols = st.columns(len(factors))
                for i, (name, points, color) in enumerate(factors):
                    with cols[i]:
                        if color == 'green':
                            st.success(f"**{name}**\n\n{points}")
                        elif color == 'red':
                            st.error(f"**{name}**\n\n{points}")
                        else:
                            st.warning(f"**{name}**\n\n{points}")
            else:
                st.info("No significant factors detected")
            
            st.divider()
            
            # --- OI vs PRICE CHART WITH SIGNALS ---
            st.markdown("### 📈 Open Interest vs Price (with Historical Signals)")
            
            if not oi_history.empty and not df_price.empty:
                # Calculate historical signals for each point
                oi_history = oi_history.copy()
                oi_history['oi_pct_change'] = oi_history['sumOpenInterest'].pct_change(periods=5) * 100
                
                # Match price data to OI timeframe
                oi_start = oi_history['timestamp'].min()
                df_price_filtered = df_price[df_price['timestamp'] >= oi_start].copy()
                
                # Merge price changes into OI history for signal calculation
                if not df_price_filtered.empty:
                    df_price_filtered['price_pct_change'] = df_price_filtered['close'].pct_change(periods=5) * 100
                    
                    # Create signal markers
                    bullish_points = []
                    bearish_points = []
                    weak_points = []
                    cap_points = []
                    
                    for i, row in oi_history.iterrows():
                        oi_chg = row.get('oi_pct_change', 0)
                        if pd.isna(oi_chg):
                            continue
                        
                        # Find closest price data
                        ts = row['timestamp']
                        price_match = df_price_filtered[df_price_filtered['timestamp'] <= ts]
                        if price_match.empty:
                            continue
                        price_chg = price_match['price_pct_change'].iloc[-1] if not pd.isna(price_match['price_pct_change'].iloc[-1]) else 0
                        
                        # Apply signal logic
                        if oi_chg > 0.5 and price_chg > 0.5:
                            bullish_points.append((ts, row['sumOpenInterest'], 'Bullish'))
                        elif oi_chg > 0.5 and price_chg < -0.5:
                            bearish_points.append((ts, row['sumOpenInterest'], 'Bearish'))
                        elif oi_chg < -0.5 and price_chg > 0.5:
                            weak_points.append((ts, row['sumOpenInterest'], 'Weak Rally'))
                        elif oi_chg < -0.5 and price_chg < -0.5:
                            cap_points.append((ts, row['sumOpenInterest'], 'Capitulation'))
                
                # Create chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # OI line
                fig.add_trace(
                    go.Scatter(
                        x=oi_history['timestamp'],
                        y=oi_history['sumOpenInterest'],
                        name="Open Interest",
                        line=dict(color='#00bcd4', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 188, 212, 0.1)'
                    ),
                    secondary_y=False
                )
                
                # Price line
                if not df_price_filtered.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df_price_filtered['timestamp'],
                            y=df_price_filtered['close'],
                            name="Price",
                            line=dict(color='#ff9800', width=2)
                        ),
                        secondary_y=True
                    )
                
                # Add signal markers
                if bullish_points:
                    fig.add_trace(go.Scatter(
                        x=[p[0] for p in bullish_points],
                        y=[p[1] for p in bullish_points],
                        mode='markers',
                        name='🟢 Bullish',
                        marker=dict(color='lime', size=10, symbol='triangle-up'),
                        hovertemplate='Bullish Signal<extra></extra>'
                    ), secondary_y=False)
                
                if bearish_points:
                    fig.add_trace(go.Scatter(
                        x=[p[0] for p in bearish_points],
                        y=[p[1] for p in bearish_points],
                        mode='markers',
                        name='🔴 Bearish',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        hovertemplate='Bearish Signal<extra></extra>'
                    ), secondary_y=False)
                
                if weak_points:
                    fig.add_trace(go.Scatter(
                        x=[p[0] for p in weak_points],
                        y=[p[1] for p in weak_points],
                        mode='markers',
                        name='🟡 Weak Rally',
                        marker=dict(color='yellow', size=8, symbol='circle'),
                        hovertemplate='Weak Rally<extra></extra>'
                    ), secondary_y=False)
                
                if cap_points:
                    fig.add_trace(go.Scatter(
                        x=[p[0] for p in cap_points],
                        y=[p[1] for p in cap_points],
                        mode='markers',
                        name='🟠 Capitulation',
                        marker=dict(color='orange', size=8, symbol='diamond'),
                        hovertemplate='Capitulation<extra></extra>'
                    ), secondary_y=False)
                
                fig.update_layout(
                    height=500,
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_yaxes(title_text="Open Interest", secondary_y=False, tickformat=".2s")
                fig.update_yaxes(title_text="Price (USD)", secondary_y=True, tickprefix="$")
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                
                # Signal count summary
                st.caption(f"📊 Historical Signals: 🟢 {len(bullish_points)} Bullish | 🔴 {len(bearish_points)} Bearish | 🟡 {len(weak_points)} Weak Rally | 🟠 {len(cap_points)} Capitulation")
            else:
                # Debug info - show which data source is missing
                oi_rows = len(oi_history) if not oi_history.empty else 0
                price_rows = len(df_price) if not df_price.empty else 0
                
                st.warning(f"📊 Data Status: OI History: {oi_rows} rows | Price Data: {price_rows} rows")
                
                if oi_rows == 0:
                    st.error("""
                    **❌ OI History Failed to Load**
                    
                    Possible causes:
                    1. SOCKS5 proxy not configured or not working
                    2. Binance API blocked (common from USA without proxy)
                    3. Symbol doesn't have futures data
                    
                    **Fix:** Check `SOCKS5_PROXY_URL` in Streamlit Secrets
                    """)
                elif price_rows == 0:
                    st.error("❌ Price data failed to load from Yahoo Finance")
            
            # --- EDUCATIONAL GUIDE ---
            with st.expander("📖 How to Read Open Interest"):
                st.markdown("""
                **Open Interest (OI)** = Total outstanding futures contracts
                
                | OI Change | Price Change | Signal | Meaning |
                |-----------|-------------|--------|---------|
                | ↑ Rising | ↑ Rising | 🟢 **Bullish** | New money entering, trend strengthening |
                | ↑ Rising | ↓ Falling | 🔴 **Bearish** | Short sellers entering aggressively |
                | ↓ Falling | ↑ Rising | 🟡 **Weak Rally** | Short covering, may not sustain |
                | ↓ Falling | ↓ Falling | 🟠 **Capitulation** | Liquidations, potential bottom |
                
                **For Long-Term Investors:**
                - 🟢 STRONG TREND = Favorable for holding/adding
                - 🟡 WEAK RALLY = Be cautious, wait for confirmation
                - 🔴 SHORT PRESSURE = Consider reducing exposure
                - 🟠 CAPITULATION = Watch for reversal opportunities
                """)
    else:
        st.info("""
        👆 **Configure settings and click 'Analyze OI'**
        
        📊 **What you'll see:**
        - Current Open Interest and trend (up to 2 years history!)
        - Funding rate and Long/Short ratio
        - Investment advisory signals
        
        ⚙️ **Setup (Streamlit Cloud):**
        1. Add `COINALYZE_API_KEY` to your Streamlit Secrets
        2. Get free API key at coinalyze.net
        3. Select a symbol and click 'Analyze OI'
        """)

# Default
if analysis_mode == "📊 Single Asset" and ('analyze_btn' not in dir() or not analyze_btn):
    st.info("""
👈 **Select mode:**
- **Single Asset**: Full analysis with zones & signals
- **Watchlist**: Quick overview of multiple assets  
- **Timeframe Compare**: Daily vs Weekly alignment
- **Open Interest Monitor**: Futures OI analysis & advisory

**Tip:** Focus on weekly charts. When both Daily AND Weekly show accumulation = high confidence entry.
    """)

