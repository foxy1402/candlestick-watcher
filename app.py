import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Pattern Watcher", page_icon="🕯️")

# --- Session State Initialization ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['BTC-USD', 'ETH-USD', 'SOL-USD']

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
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance - full history."""
    try:
        ticker = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        # Fetch full historical data
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
    except Exception as e:
        return pd.DataFrame()

def fetch_data_raw(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch data WITHOUT cache - for use in parallel threads where st.cache doesn't work."""
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
    except Exception as e:
        return pd.DataFrame()

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
    # Vectorized A/D calculation
    df['ad'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)
    df['ad_ema'] = talib.EMA(df['ad'].values, timeperiod=21)
    
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

def detect_wyckoff_fast(df: pd.DataFrame, lookback: int = 52) -> dict:
    """Simplified Wyckoff detection - optimized."""
    if len(df) < lookback:
        return {"phase": "Insufficient Data", "emoji": "⚪", "label": "N/A", "description": "Need more data", "color": "gray"}
    
    recent = df.tail(lookback)
    current_price = recent['close'].iloc[-1]
    price_high = recent['high'].max()
    price_low = recent['low'].min()
    price_range = price_high - price_low
    
    if price_range == 0:
        return {"phase": "Ranging", "emoji": "↔️", "label": "SIDEWAYS", "description": "Market consolidating", "color": "gray"}
    
    price_position = (current_price - price_low) / price_range
    ad_trend = recent['ad'].iloc[-1] - recent['ad'].iloc[0] if 'ad' in recent.columns else 0
    price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
    
    if price_position < 0.3 and ad_trend > 0:
        return {"phase": "Accumulation", "emoji": "🛒", "label": "SMART MONEY BUYING",
                "description": "Big players quietly accumulating", "color": "green"}
    elif price_position > 0.7 and ad_trend < 0:
        return {"phase": "Distribution", "emoji": "💸", "label": "SMART MONEY SELLING",
                "description": "Big players quietly selling", "color": "red"}
    elif ad_trend > 0 and price_trend > 0:
        return {"phase": "Markup", "emoji": "📈", "label": "TRENDING UP",
                "description": "Healthy uptrend with volume", "color": "green"}
    elif ad_trend < 0 and price_trend < 0:
        return {"phase": "Markdown", "emoji": "📉", "label": "TRENDING DOWN",
                "description": "Downtrend confirmed", "color": "red"}
    return {"phase": "Ranging", "emoji": "↔️", "label": "SIDEWAYS",
            "description": "Market consolidating", "color": "gray"}

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

def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate multi-period performance returns using date-based lookback."""
    if df.empty or len(df) < 2:
        return {'7d': 0, '30d': 0, '90d': 0, 'ytd': 0}
    
    # Ensure timestamp is datetime first
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    current_price = df['close'].iloc[-1]
    current_date = pd.to_datetime(df['timestamp'].iloc[-1])
    
    # 7D return (find price ~7 days ago)
    target_7d = current_date - pd.Timedelta(days=7)
    df_7d = df[df['timestamp'] <= target_7d]
    if not df_7d.empty:
        price_7d = df_7d['close'].iloc[-1]
        ret_7d = ((current_price - price_7d) / price_7d) * 100
    else:
        ret_7d = 0
    
    # 30D return
    target_30d = current_date - pd.Timedelta(days=30)
    df_30d = df[df['timestamp'] <= target_30d]
    if not df_30d.empty:
        price_30d = df_30d['close'].iloc[-1]
        ret_30d = ((current_price - price_30d) / price_30d) * 100
    else:
        ret_30d = 0
    
    # 90D return
    target_90d = current_date - pd.Timedelta(days=90)
    df_90d = df[df['timestamp'] <= target_90d]
    if not df_90d.empty:
        price_90d = df_90d['close'].iloc[-1]
        ret_90d = ((current_price - price_90d) / price_90d) * 100
    else:
        ret_90d = 0
    
    # YTD return
    current_year = current_date.year
    ytd_data = df[df['timestamp'].dt.year == current_year]
    if len(ytd_data) > 1:
        price_ytd_start = ytd_data['close'].iloc[0]
        ret_ytd = ((current_price - price_ytd_start) / price_ytd_start) * 100
    else:
        ret_ytd = 0
    
    return {'7d': ret_7d, '30d': ret_30d, '90d': ret_90d, 'ytd': ret_ytd}

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
    
    # Entry zone (around support)
    entry_low = support1
    entry_high = support1 + (pivot - support1) * 0.3
    
    return {
        'support': support1,
        'resistance': resistance1,
        'entry_low': entry_low,
        'entry_high': entry_high,
        'pivot': pivot
    }

def fetch_symbol_status_enhanced(symbol: str, interval: str, lookback: int) -> dict:
    """Fetch comprehensive investment data for a single symbol."""
    try:
        df = fetch_data_raw(symbol, interval)
        if df.empty:
            return {
                'symbol': symbol, 'status': '❓', 'phase': 'No Data', 
                'price': 0, 'change': 0, 'wyckoff': 'N/A', 'wyckoff_emoji': '❓',
                '7d': 0, '30d': 0, '90d': 0, 'ytd': 0,
                'signal_score': 0, 'levels': {}, 'action': 'NO DATA'
            }
        
        phase, _, df = analyze_ad_phase_fast(df, lookback)
        wyckoff = detect_wyckoff_fast(df, lookback)
        
        last_price = df.iloc[-1]['close']
        prev_price = df.iloc[-2]['close'] if len(df) > 1 else last_price
        pct_change = ((last_price - prev_price) / prev_price) * 100
        
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

def get_watchlist_status_parallel(symbols: list, interval: str = '1wk', lookback: int = 52) -> list:
    """Fetch watchlist status SEQUENTIALLY - yfinance has thread-safety issues with ThreadPoolExecutor."""
    results = []
    for sym in symbols:
        result = fetch_symbol_status_enhanced(sym, interval, lookback)
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
    
    # Trend direction (30 points)
    daily_trend = 1 if df_daily['close'].iloc[-1] > df_daily['close'].iloc[-20] else -1 if len(df_daily) >= 20 else 0
    weekly_trend = 1 if df_weekly['close'].iloc[-1] > df_weekly['close'].iloc[-10] else -1 if len(df_weekly) >= 10 else 0
    
    if daily_trend == weekly_trend == 1:
        trend_align = 30
    elif daily_trend == weekly_trend == -1:
        trend_align = -30
    else:
        trend_align = 0
    
    score += trend_align // 2
    factors['trend_alignment'] = trend_align
    
    # A/D momentum (30 points)
    if 'ad' in df_daily.columns and 'ad' in df_weekly.columns:
        ad_daily = df_daily['ad'].iloc[-1] - df_daily['ad'].iloc[-20] if len(df_daily) >= 20 else 0
        ad_weekly = df_weekly['ad'].iloc[-1] - df_weekly['ad'].iloc[-10] if len(df_weekly) >= 10 else 0
        
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

# --- UI ---
st.title("🕯️ Crypto Pattern Watcher")
st.caption("Long-term A/D Analysis | Multi-Asset Watchlist | Entry Signals | Simplified Wyckoff")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ["📊 Single Asset", "📋 Watchlist Dashboard", "🔄 Timeframe Compare"],
        index=0
    )
    
    st.divider()
    
    if analysis_mode == "📊 Single Asset":
        symbol = st.text_input("Symbol", value="BTC-USD").upper()
        interval = st.selectbox("Timeframe", ["1d", "1wk"], format_func=lambda x: "Daily" if x == "1d" else "Weekly")
        
        lookback_presets = {"Short (10)": 10, "Mid (26)": 26, "Long (52)": 52}
        lookback_selection = st.selectbox("Lookback", options=list(lookback_presets.keys()), index=2)
        lookback_period = lookback_presets[lookback_selection]
        
        analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
    
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
    
    else:
        symbol = st.text_input("Symbol", value="BTC-USD").upper()
        compare_btn = st.button("🔄 Compare", use_container_width=True, type="primary")

# --- Main Content ---

# Single Asset
if analysis_mode == "📊 Single Asset" and 'analyze_btn' in dir() and analyze_btn:
    with st.spinner(f"Analyzing {symbol}..."):
        df = fetch_data(symbol, interval)
    
    if not df.empty:
        df = detect_patterns_optimized(df)
        phase, color, df = analyze_ad_phase_fast(df, lookback=lookback_period)
        df = generate_signals_fast(df)
        wyckoff = detect_wyckoff_fast(df, lookback_period)
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
                hovertemplate='STRONG BUY<extra></extra>'
            ))
        
        if not strong_sells.empty:
            fig.add_trace(go.Scatter(
                x=strong_sells['timestamp'], y=strong_sells['high'] * 1.03,
                mode='markers', name='Strong Sell',
                marker=dict(color='red', size=12, symbol='star'),
                hovertemplate='STRONG SELL<extra></extra>'
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
    else:
        st.error(f"Could not load {symbol}")

# Watchlist
elif analysis_mode == "📋 Watchlist Dashboard":
    st.subheader("📋 Investment Watchlist Dashboard")
    
    if not st.session_state.watchlist:
        st.info("Watchlist empty. Add symbols via sidebar.")
    elif 'refresh_btn' in dir() and refresh_btn:
        with st.spinner("Fetching comprehensive data..."):
            data = get_watchlist_status_parallel(st.session_state.watchlist)
        
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
            df_d = fetch_data(symbol, '1d')
            df_w = fetch_data(symbol, '1wk')
        
        if not df_d.empty and not df_w.empty:
            # Calculate alignment score
            alignment = calculate_mtf_alignment(df_d.copy(), df_w.copy())
            
            # Re-analyze for display (since mtf_alignment modifies dfs)
            phase_d, _, df_d = analyze_ad_phase_fast(df_d, 26)
            phase_w, _, df_w = analyze_ad_phase_fast(df_w, 52)
            wyck_d = detect_wyckoff_fast(df_d, 26)
            wyck_w = detect_wyckoff_fast(df_w, 52)
            
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

# Default
if analysis_mode == "📊 Single Asset" and ('analyze_btn' not in dir() or not analyze_btn):
    st.info("""
👈 **Select mode:**
- **Single Asset**: Full analysis with zones & signals
- **Watchlist**: Quick overview of multiple assets  
- **Timeframe Compare**: Daily vs Weekly alignment

**Tip:** Focus on weekly charts. When both Daily AND Weekly show accumulation = high confidence entry.
    """)
