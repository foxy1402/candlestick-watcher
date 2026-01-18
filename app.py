import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

BULLISH_PATTERNS = ['HAMMER', 'MORNINGSTAR', 'ENGULFING', 'PIERCING', '3WHITESOLDIERS', 'MORNINGDOJISTAR', 'HARAMI', '3INSIDE', '3OUTSIDE']
BEARISH_PATTERNS = ['SHOOTINGSTAR', 'EVENINGSTAR', 'ENGULFING', 'DARKCLOUDCOVER', '3BLACKCROWS', 'EVENINGDOJISTAR', 'HANGINGMAN']

# --- Functions ---
@st.cache_data(ttl=300)
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
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
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TA-Lib pattern recognition and rank results."""
    candle_names = talib.get_function_groups()['Pattern Recognition']
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    
    for candle in candle_names:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)
    
    df['candlestick_pattern'] = "NO_PATTERN"
    df['candlestick_match_count'] = 0
    df['pattern_direction'] = 'neutral'
    
    for idx, row in df.iterrows():
        found = [(c, row[c]) for c in candle_names if row[c] != 0]
        if found:
            ranked = sorted(
                [(f"{n}_{'Bull' if v > 0 else 'Bear'}", PATTERN_RANKINGS.get(f"{n}_{'Bull' if v > 0 else 'Bear'}", 999), v) for n, v in found],
                key=lambda x: x[1]
            )
            df.loc[idx, 'candlestick_pattern'] = ranked[0][0]
            df.loc[idx, 'candlestick_match_count'] = len(ranked)
            df.loc[idx, 'pattern_direction'] = 'bullish' if ranked[0][2] > 0 else 'bearish'
    
    df['pattern_display'] = df['candlestick_pattern'].str.replace('NO_PATTERN|CDL|_Bull|_Bear', '', regex=True)
    return df

def analyze_ad_phase(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """Calculate Chaikin A/D and determine market phase with historical phases."""
    df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ad_ema'] = talib.EMA(df['ad'], timeperiod=21)
    df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
    
    # Calculate rolling price and A/D changes for historical phase detection
    df['price_change'] = df['close'].diff(lookback)
    df['ad_change'] = df['ad'].diff(lookback)
    
    # Determine phase for each bar
    conditions = [
        (df['price_change'] < 0) & (df['ad_change'] > 0),  # Accumulation
        (df['price_change'] > 0) & (df['ad_change'] < 0),  # Distribution
        (df['ad_change'] > 0),  # Uptrend
        (df['ad_change'] < 0),  # Downtrend
    ]
    choices = ['accumulation', 'distribution', 'uptrend', 'downtrend']
    df['phase'] = np.select(conditions, choices, default='neutral')
    
    # Current phase analysis
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

def detect_wyckoff_phase(df: pd.DataFrame, lookback: int = 52) -> dict:
    """Simplified Wyckoff phase detection for beginners."""
    if len(df) < lookback:
        return {"phase": "Insufficient Data", "emoji": "⚪", "description": "Need more historical data"}
    
    recent = df.tail(lookback)
    current_price = recent['close'].iloc[-1]
    price_high = recent['high'].max()
    price_low = recent['low'].min()
    price_range = price_high - price_low
    
    # Volume analysis
    avg_volume = recent['volume'].mean()
    recent_volume = df.tail(5)['volume'].mean()
    volume_spike = recent_volume > avg_volume * 1.5
    
    # Price position in range
    price_position = (current_price - price_low) / price_range if price_range > 0 else 0.5
    
    # A/D trend
    ad_trend = recent['ad'].iloc[-1] - recent['ad'].iloc[0]
    price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
    
    # Simplified Wyckoff Detection
    if price_position < 0.3 and ad_trend > 0:
        # Near lows but money flowing in
        if volume_spike and price_trend > 0:
            return {"phase": "Spring (Shakeout)", "emoji": "⚡", "label": "POTENTIAL BUY", 
                    "description": "Price shook out weak hands, smart money buying", "color": "green"}
        return {"phase": "Accumulation", "emoji": "🛒", "label": "SMART MONEY BUYING",
                "description": "Big players quietly accumulating at low prices", "color": "green"}
    
    elif price_position > 0.7 and ad_trend < 0:
        # Near highs but money flowing out
        if volume_spike and price_trend < 0:
            return {"phase": "UTAD (Bull Trap)", "emoji": "💀", "label": "POTENTIAL SELL",
                    "description": "False breakout, smart money distributing", "color": "red"}
        return {"phase": "Distribution", "emoji": "💸", "label": "SMART MONEY SELLING",
                "description": "Big players quietly selling at high prices", "color": "red"}
    
    elif ad_trend > 0 and price_trend > 0:
        return {"phase": "Markup", "emoji": "📈", "label": "TRENDING UP",
                "description": "Healthy uptrend with volume confirmation", "color": "green"}
    
    elif ad_trend < 0 and price_trend < 0:
        return {"phase": "Markdown", "emoji": "📉", "label": "TRENDING DOWN",
                "description": "Downtrend with volume confirmation", "color": "red"}
    
    return {"phase": "Ranging", "emoji": "↔️", "label": "SIDEWAYS",
            "description": "Market consolidating, wait for direction", "color": "gray"}

def generate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Combine A/D phase with patterns to generate entry signals."""
    df['signal'] = 'none'
    df['signal_strength'] = 'none'
    
    for idx, row in df.iterrows():
        phase = row.get('phase', 'neutral')
        pattern = row.get('candlestick_pattern', 'NO_PATTERN')
        direction = row.get('pattern_direction', 'neutral')
        
        if pattern == 'NO_PATTERN':
            continue
            
        # Strong signals: Phase + matching pattern direction
        if phase == 'accumulation' and direction == 'bullish':
            df.loc[idx, 'signal'] = 'strong_buy'
            df.loc[idx, 'signal_strength'] = 'STRONG BUY ⭐🟢'
        elif phase == 'distribution' and direction == 'bearish':
            df.loc[idx, 'signal'] = 'strong_sell'
            df.loc[idx, 'signal_strength'] = 'STRONG SELL ⭐🔴'
        # Weak signals: Trend + matching pattern
        elif phase == 'uptrend' and direction == 'bullish':
            df.loc[idx, 'signal'] = 'weak_buy'
            df.loc[idx, 'signal_strength'] = 'BUY 🟢'
        elif phase == 'downtrend' and direction == 'bearish':
            df.loc[idx, 'signal'] = 'weak_sell'
            df.loc[idx, 'signal_strength'] = 'SELL 🔴'
        # Contrarian signals (caution)
        elif phase == 'distribution' and direction == 'bullish':
            df.loc[idx, 'signal'] = 'caution_buy'
            df.loc[idx, 'signal_strength'] = 'CAUTION BUY ⚠️'
        elif phase == 'accumulation' and direction == 'bearish':
            df.loc[idx, 'signal'] = 'caution_sell'
            df.loc[idx, 'signal_strength'] = 'CAUTION SELL ⚠️'
    
    return df

def get_phase_zones(df: pd.DataFrame) -> list:
    """Identify contiguous phase zones for chart overlay."""
    zones = []
    if df.empty or 'phase' not in df.columns:
        return zones
    
    current_phase = None
    zone_start = None
    
    for idx, row in df.iterrows():
        phase = row['phase']
        if phase in ['accumulation', 'distribution']:
            if phase != current_phase:
                if current_phase is not None:
                    zones.append({
                        'phase': current_phase,
                        'start': zone_start,
                        'end': df.loc[idx-1, 'timestamp'] if idx > 0 else row['timestamp'],
                        'start_price': df.loc[zone_start_idx, 'low'],
                        'end_price': df.loc[idx-1, 'high'] if idx > 0 else row['high']
                    })
                current_phase = phase
                zone_start = row['timestamp']
                zone_start_idx = idx
        else:
            if current_phase is not None:
                zones.append({
                    'phase': current_phase,
                    'start': zone_start,
                    'end': df.loc[idx-1, 'timestamp'] if idx > 0 else row['timestamp'],
                    'start_price': df.loc[zone_start_idx, 'low'],
                    'end_price': df.loc[idx-1, 'high'] if idx > 0 else row['high']
                })
                current_phase = None
    
    # Close final zone if exists
    if current_phase is not None:
        zones.append({
            'phase': current_phase,
            'start': zone_start,
            'end': df.iloc[-1]['timestamp'],
            'start_price': df.loc[zone_start_idx, 'low'],
            'end_price': df.iloc[-1]['high']
        })
    
    return zones

def get_watchlist_status(symbols: list, interval: str = '1wk', lookback: int = 52) -> list:
    """Get A/D status for all watchlist symbols."""
    results = []
    for symbol in symbols:
        try:
            df = fetch_data(symbol, interval)
            if df.empty:
                results.append({'symbol': symbol, 'status': '❓', 'phase': 'No Data', 'price': 0, 'change': 0})
                continue
            
            phase, color, df = analyze_ad_phase(df, lookback)
            wyckoff = detect_wyckoff_phase(df, lookback)
            
            last_price = df.iloc[-1]['close']
            prev_price = df.iloc[-2]['close'] if len(df) > 1 else last_price
            pct_change = ((last_price - prev_price) / prev_price) * 100
            
            status_map = {
                'accumulation': '🟢',
                'distribution': '🔴',
                'uptrend': '📈',
                'downtrend': '📉',
                'neutral': '⚪'
            }
            
            results.append({
                'symbol': symbol,
                'status': status_map.get(phase, '⚪'),
                'phase': phase.title(),
                'wyckoff': wyckoff['label'],
                'wyckoff_emoji': wyckoff['emoji'],
                'price': last_price,
                'change': pct_change
            })
        except Exception as e:
            results.append({'symbol': symbol, 'status': '❌', 'phase': 'Error', 'price': 0, 'change': 0})
    
    return results

# --- UI ---
st.title("🕯️ Crypto Pattern Watcher")
st.caption("Long-term A/D Analysis | Multi-Asset Watchlist | Entry Signals | Simplified Wyckoff")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Analysis Mode Selection
    analysis_mode = st.radio(
        "Analysis Mode",
        ["📊 Single Asset", "📋 Watchlist Dashboard", "🔄 Timeframe Compare"],
        index=0
    )
    
    st.divider()
    
    if analysis_mode == "📊 Single Asset":
        symbol = st.text_input("Symbol", value="BTC-USD").upper()
        interval = st.selectbox("Timeframe", ["1d", "1wk"], format_func=lambda x: "Daily" if x == "1d" else "Weekly")
        
        lookback_presets = {
            "Short-Term (10 bars)": 10,
            "Mid-Term (26 bars)": 26,
            "Long-Term (52 bars)": 52
        }
        lookback_selection = st.selectbox(
            "Lookback Period",
            options=list(lookback_presets.keys()),
            index=2,
            help="How many bars to compare for divergence detection."
        )
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
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.watchlist = []
                st.rerun()
        
        st.caption("Current watchlist:")
        for i, sym in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([3, 1])
            col1.write(sym)
            if col2.button("❌", key=f"del_{i}"):
                st.session_state.watchlist.remove(sym)
                st.rerun()
        
        refresh_btn = st.button("🔄 Refresh Dashboard", use_container_width=True, type="primary")
    
    else:  # Timeframe Compare
        symbol = st.text_input("Symbol", value="BTC-USD").upper()
        compare_btn = st.button("🔄 Compare Timeframes", use_container_width=True, type="primary")

# --- Main Content ---

# Single Asset Analysis
if analysis_mode == "📊 Single Asset" and 'analyze_btn' in dir() and analyze_btn:
    with st.spinner(f"Analyzing {symbol}..."):
        df = fetch_data(symbol, interval)
    
    if not df.empty:
        # Run all analyses
        df = detect_patterns(df)
        phase, color, df = analyze_ad_phase(df, lookback=lookback_period)
        df = generate_entry_signals(df)
        wyckoff = detect_wyckoff_phase(df, lookback_period)
        zones = get_phase_zones(df)
        
        # Header metrics
        last = df.iloc[-1]
        prev = df.iloc[-2]
        pct_change = ((last['close'] - prev['close']) / prev['close']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{symbol}", f"${last['close']:,.2f}", f"{pct_change:+.2f}%")
        col2.metric("Market Phase", phase.title(), delta=None)
        col3.metric("Wyckoff", wyckoff['phase'], delta=None)
        col4.metric("Outlook", wyckoff['label'], delta=None)
        
        # Wyckoff Explanation Card
        with st.expander(f"{wyckoff['emoji']} **{wyckoff['phase']}** - What This Means for You", expanded=True):
            st.markdown(f"""
            ### {wyckoff['label']}
            
            {wyckoff['description']}
            
            **For Long-Term Investors:**
            """)
            if wyckoff['color'] == 'green':
                st.success("✅ This is generally a favorable environment for accumulating positions.")
            elif wyckoff['color'] == 'red':
                st.error("⚠️ Consider taking profits or waiting for better entry points.")
            else:
                st.info("⏳ Market is consolidating. Wait for clearer direction before acting.")
        
        # Main Chart with Phase Zones
        st.subheader("📈 Price Chart with A/D Zones")
        fig = go.Figure()
        
        # Add phase zones as background shapes
        for zone in zones:
            zone_color = 'rgba(0, 255, 0, 0.1)' if zone['phase'] == 'accumulation' else 'rgba(255, 0, 0, 0.1)'
            fig.add_vrect(
                x0=zone['start'], x1=zone['end'],
                fillcolor=zone_color,
                layer="below",
                line_width=0,
                annotation_text=zone['phase'].title(),
                annotation_position="top left",
                annotation_font_size=10
            )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name=symbol, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))
        
        # Entry signal markers
        strong_buys = df[df['signal'] == 'strong_buy']
        strong_sells = df[df['signal'] == 'strong_sell']
        
        if not strong_buys.empty:
            fig.add_trace(go.Scatter(
                x=strong_buys['timestamp'], y=strong_buys['low'] * 0.97,
                mode='markers+text', name='Strong Buy',
                marker=dict(color='lime', size=15, symbol='star'),
                text='⭐', textposition='bottom center',
                hovertemplate='%{x}<br>STRONG BUY<br>Pattern: %{customdata}<extra></extra>',
                customdata=strong_buys['candlestick_pattern']
            ))
        
        if not strong_sells.empty:
            fig.add_trace(go.Scatter(
                x=strong_sells['timestamp'], y=strong_sells['high'] * 1.03,
                mode='markers+text', name='Strong Sell',
                marker=dict(color='red', size=15, symbol='star'),
                text='⭐', textposition='top center',
                hovertemplate='%{x}<br>STRONG SELL<br>Pattern: %{customdata}<extra></extra>',
                customdata=strong_sells['candlestick_pattern']
            ))
        
        fig.update_layout(
            height=500, xaxis_rangeslider_visible=False,
            title=f"{symbol} - {interval.upper()} Chart with A/D Zones",
            dragmode='pan', template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        # A/D Line Chart
        st.subheader("💰 Money Flow (A/D Line)")
        ad_fig = go.Figure()
        ad_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ad'], name='A/D Line', line=dict(color='orange', width=2)))
        ad_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ad_ema'], name='EMA 21', line=dict(color='yellow', dash='dot')))
        ad_fig.update_layout(height=300, template='plotly_dark', title="Accumulation/Distribution Line")
        st.plotly_chart(ad_fig, use_container_width=True, config={'scrollZoom': True})
        
        # Signal History Table
        st.subheader("📊 Recent Entry Signals")
        signals = df[df['signal'] != 'none'].tail(10).sort_values('timestamp', ascending=False)
        if not signals.empty:
            display_df = signals[['timestamp', 'close', 'candlestick_pattern', 'phase', 'signal_strength']].copy()
            display_df.columns = ['Date', 'Price', 'Pattern', 'Phase', 'Signal']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No entry signals detected in the visible range.")
        
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)
    else:
        st.error(f"Could not load data for {symbol}.")

# Watchlist Dashboard
elif analysis_mode == "📋 Watchlist Dashboard":
    st.subheader("📋 Watchlist Dashboard")
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add symbols using the sidebar.")
    else:
        if 'refresh_btn' in dir() and refresh_btn:
            with st.spinner("Fetching data for all assets..."):
                watchlist_data = get_watchlist_status(st.session_state.watchlist)
            
            # Summary cards
            accum_count = sum(1 for w in watchlist_data if w['phase'] == 'Accumulation')
            distrib_count = sum(1 for w in watchlist_data if w['phase'] == 'Distribution')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 In Accumulation", accum_count)
            col2.metric("🔴 In Distribution", distrib_count)
            col3.metric("Total Assets", len(watchlist_data))
            
            st.divider()
            
            # Watchlist table
            for item in watchlist_data:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
                    col1.markdown(f"### {item['status']} {item['symbol']}")
                    col2.metric("Price", f"${item['price']:,.2f}" if item['price'] > 0 else "N/A")
                    col3.metric("Phase", item['phase'])
                    col4.metric("Wyckoff", f"{item.get('wyckoff_emoji', '')} {item.get('wyckoff', 'N/A')}")
                    change_color = "green" if item['change'] > 0 else "red"
                    col5.markdown(f"<span style='color:{change_color}'>{item['change']:+.2f}%</span>", unsafe_allow_html=True)
                st.divider()
        else:
            st.info("👆 Click 'Refresh Dashboard' to load watchlist data.")

# Timeframe Comparison
elif analysis_mode == "🔄 Timeframe Compare":
    st.subheader(f"🔄 Timeframe Comparison: {symbol}")
    
    if 'compare_btn' in dir() and compare_btn:
        with st.spinner("Fetching daily and weekly data..."):
            df_daily = fetch_data(symbol, '1d')
            df_weekly = fetch_data(symbol, '1wk')
        
        if not df_daily.empty and not df_weekly.empty:
            # Analyze both timeframes
            phase_daily, _, df_daily = analyze_ad_phase(df_daily, 26)
            phase_weekly, _, df_weekly = analyze_ad_phase(df_weekly, 52)
            wyckoff_daily = detect_wyckoff_phase(df_daily, 26)
            wyckoff_weekly = detect_wyckoff_phase(df_weekly, 52)
            
            # Alignment check
            st.subheader("📊 Timeframe Alignment")
            
            daily_bullish = phase_daily in ['accumulation', 'uptrend']
            weekly_bullish = phase_weekly in ['accumulation', 'uptrend']
            
            if daily_bullish and weekly_bullish:
                st.success("✅ **STRONG BUY ZONE** - Both Daily and Weekly are bullish. High confidence for long-term entry.")
            elif not daily_bullish and not weekly_bullish:
                st.error("🔴 **STRONG SELL ZONE** - Both Daily and Weekly are bearish. Consider waiting or taking profits.")
            else:
                st.warning("⚠️ **CAUTION** - Timeframes are not aligned. Wait for confirmation before acting.")
            
            # Side by side metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Daily")
                st.metric("Phase", phase_daily.title())
                st.metric("Wyckoff", f"{wyckoff_daily['emoji']} {wyckoff_daily['label']}")
                st.caption(wyckoff_daily['description'])
            
            with col2:
                st.markdown("### 📆 Weekly")
                st.metric("Phase", phase_weekly.title())
                st.metric("Wyckoff", f"{wyckoff_weekly['emoji']} {wyckoff_weekly['label']}")
                st.caption(wyckoff_weekly['description'])
            
            # Stacked charts
            st.subheader("📈 Daily vs Weekly Charts")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                               subplot_titles=("Daily", "Weekly"),
                               vertical_spacing=0.1)
            
            # Daily candlestick
            fig.add_trace(go.Candlestick(
                x=df_daily.tail(90)['timestamp'], 
                open=df_daily.tail(90)['open'], high=df_daily.tail(90)['high'],
                low=df_daily.tail(90)['low'], close=df_daily.tail(90)['close'],
                name='Daily', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            # Weekly candlestick
            fig.add_trace(go.Candlestick(
                x=df_weekly.tail(52)['timestamp'],
                open=df_weekly.tail(52)['open'], high=df_weekly.tail(52)['high'],
                low=df_weekly.tail(52)['low'], close=df_weekly.tail(52)['close'],
                name='Weekly', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=2, col=1)
            
            fig.update_layout(height=700, template='plotly_dark', showlegend=False)
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        else:
            st.error(f"Could not load data for {symbol}.")
    else:
        st.info("👆 Click 'Compare Timeframes' to analyze Daily vs Weekly.")

# Default state
if analysis_mode == "📊 Single Asset" and ('analyze_btn' not in dir() or not analyze_btn):
    st.info("""
    👈 **Select an analysis mode from the sidebar:**
    
    - **Single Asset**: Deep analysis of one symbol with A/D zones, Wyckoff phases, and entry signals
    - **Watchlist Dashboard**: Quick overview of multiple assets at a glance
    - **Timeframe Compare**: See if Daily and Weekly charts are aligned
    
    **Pro Tip for Long-Term Investors:**
    Focus on weekly charts and look for Accumulation phases. When both Daily AND Weekly show accumulation, it's often a high-confidence entry zone.
    """)
