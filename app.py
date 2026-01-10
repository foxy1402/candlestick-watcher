import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Pattern Watcher")

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

PATTERN_DESCRIPTIONS = {
    "ENGULFING": "Reversal pattern: candle body overshadows the previous one.",
    "HAMMER": "Bullish reversal: small body near top, long lower shadow.",
    "HANGINGMAN": "Bearish reversal: small body near top, long lower shadow.",
    "DOJI": "Indecision: open and close are virtually the same.",
    "MORNINGSTAR": "Bullish 3-candle reversal: long red, small body, long green.",
    "EVENINGSTAR": "Bearish 3-candle reversal: long green, small body, long red.",
    "MARUBOZU": "Strong trend: candle with no shadows, just a full body.",
    "SHOOTINGSTAR": "Bearish reversal: small body near bottom, long upper shadow."
}

# --- Functions ---
@st.cache_data(ttl=300)
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        ticker = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        df = yf.download(ticker, period="1y", interval=interval, progress=False)
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
    
    for idx, row in df.iterrows():
        found = [(c, row[c]) for c in candle_names if row[c] != 0]
        if found:
            ranked = sorted(
                [(f"{n}_{'Bull' if v > 0 else 'Bear'}", PATTERN_RANKINGS.get(f"{n}_{'Bull' if v > 0 else 'Bear'}", 999)) for n, v in found],
                key=lambda x: x[1]
            )
            df.loc[idx, 'candlestick_pattern'] = ranked[0][0]
            df.loc[idx, 'candlestick_match_count'] = len(ranked)
    
    df['pattern_display'] = df['candlestick_pattern'].str.replace('NO_PATTERN|CDL|_Bull|_Bear', '', regex=True)
    return df

def analyze_ad_phase(df: pd.DataFrame) -> tuple:
    """Calculate Chaikin A/D and determine market phase."""
    df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ad_ema'] = talib.EMA(df['ad'], timeperiod=21)
    
    recent = df.tail(20)
    price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
    ad_change = recent['ad'].iloc[-1] - recent['ad'].iloc[0]
    
    if price_change < 0 and ad_change > 0:
        return "⚠️ ACCUMULATION (Price ↓, Money Flow ↑)", "green", df
    elif price_change > 0 and ad_change < 0:
        return "⚠️ DISTRIBUTION (Price ↑, Money Flow ↓)", "red", df
    elif ad_change > 0:
        return "Uptrend (Volume Confirmed)", "green", df
    elif ad_change < 0:
        return "Downtrend (Volume Confirmed)", "red", df
    return "Neutral", "gray", df

# --- UI ---
st.title("🕯️ Crypto Pattern Watcher")
st.caption("Real-time data via Yahoo Finance | 50+ candlestick patterns | Accumulation/Distribution analysis")

# Session State
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol", value="BTC-USD").upper()
    interval = st.selectbox("Timeframe", ["1d", "1wk"], format_func=lambda x: "1 Day" if x == "1d" else "1 Week")
    
    st.divider()
    col1, col2 = st.columns(2)
    scan_patterns = col1.button("🔍 Scan Patterns", use_container_width=True)
    analyze_phase = col2.button("📊 Analyze Phase", use_container_width=True)

# Main Logic
if scan_patterns:
    with st.spinner(f"Fetching {symbol}..."):
        st.session_state.df = fetch_data(symbol, interval)
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.session_state.df = detect_patterns(st.session_state.df)
        st.success("Pattern scan complete!")

if analyze_phase:
    if st.session_state.df is None or st.session_state.df.empty:
        with st.spinner(f"Fetching {symbol}..."):
            st.session_state.df = fetch_data(symbol, interval)
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        phase_msg, phase_color, st.session_state.df = analyze_ad_phase(st.session_state.df)
        st.success("Phase analysis complete!")

# Display Results
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    last = df.iloc[-1]
    
    # Metrics
    st.metric(f"{symbol} Price", f"${last['close']:,.2f}", f"{((last['close'] - df.iloc[-2]['close'])/df.iloc[-2]['close'])*100:.2f}%")
    
    # Candlestick Chart
    fig = go.Figure(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=symbol))
    
    if 'candlestick_pattern' in df.columns:
        patterns = df[df['candlestick_pattern'] != 'NO_PATTERN']
        if not patterns.empty:
            fig.add_trace(go.Scatter(x=patterns['timestamp'], y=patterns['high'] * 1.01, mode='markers',
                                     marker=dict(symbol='triangle-down', size=12, color='orange'), name='Pattern', hovertext=patterns['candlestick_pattern']))
    
    fig.update_layout(height=450, xaxis_rangeslider_visible=False, title=f"{symbol} Chart")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for organized content
    tab1, tab2, tab3 = st.tabs(["📋 Patterns", "📊 A/D Analysis", "📄 Raw Data"])
    
    with tab1:
        if 'candlestick_pattern' in df.columns:
            patterns = df[df['candlestick_pattern'] != 'NO_PATTERN']
            if not patterns.empty:
                recent = patterns.tail(10).sort_values('timestamp', ascending=False)
                st.dataframe(recent[['timestamp', 'close', 'candlestick_pattern']].rename(columns={'candlestick_pattern': 'Pattern'}), use_container_width=True)
                
                latest = patterns.iloc[-1]
                base_name = latest['pattern_display']
                desc = PATTERN_DESCRIPTIONS.get(base_name, "Pattern detected.")
                st.info(f"**{latest['candlestick_pattern']}** on {latest['timestamp'].strftime('%Y-%m-%d')}: {desc}")
            else:
                st.info("No patterns detected. Try a different timeframe.")
        else:
            st.warning("Click 'Scan Patterns' first.")
    
    with tab2:
        if 'ad' in df.columns:
            recent = df.tail(20)
            price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
            ad_change = recent['ad'].iloc[-1] - recent['ad'].iloc[0]
            
            if price_change < 0 and ad_change > 0:
                st.success("⚠️ **ACCUMULATION** - Price falling but money flowing in (divergence)")
            elif price_change > 0 and ad_change < 0:
                st.error("⚠️ **DISTRIBUTION** - Price rising but money flowing out (divergence)")
            elif ad_change > 0:
                st.success("📈 **Uptrend** confirmed by volume")
            else:
                st.error("📉 **Downtrend** confirmed by volume")
            
            ad_fig = go.Figure()
            ad_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ad'], name='A/D Line', line=dict(color='orange')))
            ad_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ad_ema'], name='EMA 21', line=dict(color='yellow', dash='dot')))
            ad_fig.update_layout(title="Chaikin A/D Line", height=300)
            st.plotly_chart(ad_fig, use_container_width=True)
        else:
            st.warning("Click 'Analyze Phase' first.")
    
    with tab3:
        st.dataframe(df, use_container_width=True)
else:
    st.info("👈 Enter a symbol and click 'Scan Patterns' or 'Analyze Phase' to start.")
