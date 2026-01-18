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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data_light(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch for watchlist - same as full fetch, cached separately."""
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
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except:
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

def fetch_symbol_status(symbol: str, interval: str, lookback: int) -> dict:
    """Fetch status for a single symbol - used in parallel."""
    try:
        df = fetch_data_light(symbol, interval)
        if df.empty:
            return {'symbol': symbol, 'status': '❓', 'phase': 'No Data', 'price': 0, 'change': 0, 'wyckoff': 'N/A', 'wyckoff_emoji': '❓'}
        
        phase, _, df = analyze_ad_phase_fast(df, lookback)
        wyckoff = detect_wyckoff_fast(df, lookback)
        
        last_price = df.iloc[-1]['close']
        prev_price = df.iloc[-2]['close'] if len(df) > 1 else last_price
        pct_change = ((last_price - prev_price) / prev_price) * 100
        
        status_map = {'accumulation': '🟢', 'distribution': '🔴', 'uptrend': '📈', 'downtrend': '📉', 'neutral': '⚪'}
        
        return {
            'symbol': symbol,
            'status': status_map.get(phase, '⚪'),
            'phase': phase.title(),
            'wyckoff': wyckoff['label'],
            'wyckoff_emoji': wyckoff['emoji'],
            'price': last_price,
            'change': pct_change
        }
    except:
        return {'symbol': symbol, 'status': '❌', 'phase': 'Error', 'price': 0, 'change': 0, 'wyckoff': 'N/A', 'wyckoff_emoji': '❌'}

def get_watchlist_status_parallel(symbols: list, interval: str = '1wk', lookback: int = 52) -> list:
    """Fetch watchlist status in parallel for speed."""
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_symbol_status, sym, interval, lookback): sym for sym in symbols}
        for future in as_completed(futures):
            results.append(future.result())
    # Sort to maintain original order
    symbol_order = {sym: i for i, sym in enumerate(symbols)}
    results.sort(key=lambda x: symbol_order.get(x['symbol'], 999))
    return results

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
    st.subheader("📋 Watchlist Dashboard")
    
    if not st.session_state.watchlist:
        st.info("Watchlist empty. Add symbols via sidebar.")
    elif 'refresh_btn' in dir() and refresh_btn:
        with st.spinner("Fetching data (parallel)..."):
            data = get_watchlist_status_parallel(st.session_state.watchlist)
        
        accum = sum(1 for w in data if w['phase'] == 'Accumulation')
        distrib = sum(1 for w in data if w['phase'] == 'Distribution')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Accumulation", accum)
        col2.metric("🔴 Distribution", distrib)
        col3.metric("Total", len(data))
        
        st.divider()
        
        for item in data:
            cols = st.columns([2, 1, 2, 2, 1])
            cols[0].markdown(f"**{item['status']} {item['symbol']}**")
            cols[1].write(f"${item['price']:,.0f}" if item['price'] > 0 else "N/A")
            cols[2].write(item['phase'])
            cols[3].write(f"{item['wyckoff_emoji']} {item['wyckoff']}")
            color = "green" if item['change'] > 0 else "red"
            cols[4].markdown(f"<span style='color:{color}'>{item['change']:+.1f}%</span>", unsafe_allow_html=True)
            st.divider()
    else:
        st.info("👆 Click 'Refresh' to load data")

# Timeframe Compare
elif analysis_mode == "🔄 Timeframe Compare":
    st.subheader(f"🔄 {symbol} - Daily vs Weekly")
    
    if 'compare_btn' in dir() and compare_btn:
        with st.spinner("Loading..."):
            df_d = fetch_data(symbol, '1d')
            df_w = fetch_data(symbol, '1wk')
        
        if not df_d.empty and not df_w.empty:
            phase_d, _, df_d = analyze_ad_phase_fast(df_d, 26)
            phase_w, _, df_w = analyze_ad_phase_fast(df_w, 52)
            wyck_d = detect_wyckoff_fast(df_d, 26)
            wyck_w = detect_wyckoff_fast(df_w, 52)
            
            daily_bull = phase_d in ['accumulation', 'uptrend']
            weekly_bull = phase_w in ['accumulation', 'uptrend']
            
            if daily_bull and weekly_bull:
                st.success("✅ **STRONG BUY ZONE** - Both timeframes bullish")
            elif not daily_bull and not weekly_bull:
                st.error("🔴 **STRONG SELL ZONE** - Both bearish")
            else:
                st.warning("⚠️ **CAUTION** - Timeframes not aligned")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📅 Daily")
                st.metric("Phase", phase_d.title())
                st.caption(wyck_d['description'])
            with col2:
                st.markdown("### 📆 Weekly")
                st.metric("Phase", phase_w.title())
                st.caption(wyck_w['description'])
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Daily", "Weekly"), vertical_spacing=0.12)
            
            fig.add_trace(go.Candlestick(
                x=df_d['timestamp'], open=df_d['open'], high=df_d['high'],
                low=df_d['low'], close=df_d['close'], name='Daily',
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            fig.add_trace(go.Candlestick(
                x=df_w['timestamp'], open=df_w['open'], high=df_w['high'],
                low=df_w['low'], close=df_w['close'], name='Weekly',
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_dark', showlegend=False)
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
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
