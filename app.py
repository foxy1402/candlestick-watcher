
import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
import plotly.graph_objects as go
from itertools import compress

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Pattern Watcher")

# --- Constants & Pattern Info ---
PATTERN_RANKINGS = {
    "CDL3LINESTRIKE_Bull": 1, "CDL3LINESTRIKE_Bear": 2,
    "CDL3BLACKCROWS_Bull": 3, "CDL3BLACKCROWS_Bear": 3,
    "CDLEVENINGSTAR_Bull": 4, "CDLEVENINGSTAR_Bear": 4,
    "CDLTASUKIGAP_Bull": 5, "CDLTASUKIGAP_Bear": 5,
    "CDLINVERTEDHAMMER_Bull": 6, "CDLINVERTEDHAMMER_Bear": 6,
    "CDLMATCHINGLOW_Bull": 7, "CDLMATCHINGLOW_Bear": 7,
    "CDLABANDONEDBABY_Bull": 8, "CDLABANDONEDBABY_Bear": 8,
    "CDLBREAKAWAY_Bull": 10, "CDLBREAKAWAY_Bear": 10,
    "CDLMORNINGSTAR_Bull": 12, "CDLMORNINGSTAR_Bear": 12,
    "CDLPIERCING_Bull": 13, "CDLPIERCING_Bear": 13,
    "CDLSTICKSANDWICH_Bull": 14, "CDLSTICKSANDWICH_Bear": 14,
    "CDLTHRUSTING_Bull": 15, "CDLTHRUSTING_Bear": 15,
    "CDLINNECK_Bull": 17, "CDLINNECK_Bear": 17,
    "CDL3INSIDE_Bull": 20, "CDL3INSIDE_Bear": 56,
    "CDLHOMINGPIGEON_Bull": 21, "CDLHOMINGPIGEON_Bear": 21,
    "CDLDARKCLOUDCOVER_Bull": 22, "CDLDARKCLOUDCOVER_Bear": 22,
    "CDLIDENTICAL3CROWS_Bull": 24, "CDLIDENTICAL3CROWS_Bear": 24,
    "CDLMORNINGDOJISTAR_Bull": 25, "CDLMORNINGDOJISTAR_Bear": 25,
    "CDLXSIDEGAP3METHODS_Bull": 27, "CDLXSIDEGAP3METHODS_Bear": 26,
    "CDLTRISTAR_Bull": 28, "CDLTRISTAR_Bear": 76,
    "CDLGAPSIDESIDEWHITE_Bull": 46, "CDLGAPSIDESIDEWHITE_Bear": 29,
    "CDLEVENINGDOJISTAR_Bull": 30, "CDLEVENINGDOJISTAR_Bear": 30,
    "CDL3WHITESOLDIERS_Bull": 32, "CDL3WHITESOLDIERS_Bear": 32,
    "CDLONNECK_Bull": 33, "CDLONNECK_Bear": 33,
    "CDL3OUTSIDE_Bull": 34, "CDL3OUTSIDE_Bear": 39,
    "CDLRICKSHAWMAN_Bull": 35, "CDLRICKSHAWMAN_Bear": 35,
    "CDLSEPARATINGLINES_Bull": 36, "CDLSEPARATINGLINES_Bear": 40,
    "CDLLONGLEGGEDDOJI_Bull": 37, "CDLLONGLEGGEDDOJI_Bear": 37,
    "CDLHARAMI_Bull": 38, "CDLHARAMI_Bear": 72,
    "CDLLADDERBOTTOM_Bull": 41, "CDLLADDERBOTTOM_Bear": 41,
    "CDLCLOSINGMARUBOZU_Bull": 70, "CDLCLOSINGMARUBOZU_Bear": 43,
    "CDLTAKURI_Bull": 47, "CDLTAKURI_Bear": 47,
    "CDLDOJISTAR_Bull": 49, "CDLDOJISTAR_Bear": 51,
    "CDLHARAMICROSS_Bull": 50, "CDLHARAMICROSS_Bear": 80,
    "CDLADVANCEBLOCK_Bull": 54, "CDLADVANCEBLOCK_Bear": 54,
    "CDLSHOOTINGSTAR_Bull": 55, "CDLSHOOTINGSTAR_Bear": 55,
    "CDLMARUBOZU_Bull": 71, "CDLMARUBOZU_Bear": 57,
    "CDLUNIQUE3RIVER_Bull": 60, "CDLUNIQUE3RIVER_Bear": 60,
    "CDL2CROWS_Bull": 61, "CDL2CROWS_Bear": 61,
    "CDLBELTHOLD_Bull": 62, "CDLBELTHOLD_Bear": 63,
    "CDLHAMMER_Bull": 65, "CDLHAMMER_Bear": 65,
    "CDLHIGHWAVE_Bull": 67, "CDLHIGHWAVE_Bear": 67,
    "CDLSPINNINGTOP_Bull": 69, "CDLSPINNINGTOP_Bear": 73,
    "CDLUPSIDEGAP2CROWS_Bull": 74, "CDLUPSIDEGAP2CROWS_Bear": 74,
    "CDLGRAVESTONEDOJI_Bull": 77, "CDLGRAVESTONEDOJI_Bear": 77,
    "CDLHIKKAKEMOD_Bull": 82, "CDLHIKKAKEMOD_Bear": 81,
    "CDLHIKKAKE_Bull": 85, "CDLHIKKAKE_Bear": 83,
    "CDLENGULFING_Bull": 84, "CDLENGULFING_Bear": 91,
    "CDLMATHOLD_Bull": 86, "CDLMATHOLD_Bear": 86,
    "CDLHANGINGMAN_Bull": 87, "CDLHANGINGMAN_Bear": 87,
    "CDLRISEFALL3METHODS_Bull": 94, "CDLRISEFALL3METHODS_Bear": 89,
    "CDLKICKING_Bull": 96, "CDLKICKING_Bear": 102,
    "CDLDRAGONFLYDOJI_Bull": 98, "CDLDRAGONFLYDOJI_Bear": 98,
    "CDLCONCEALBABYSWALL_Bull": 101, "CDLCONCEALBABYSWALL_Bear": 101,
    "CDL3STARSINSOUTH_Bull": 103, "CDL3STARSINSOUTH_Bear": 103,
    "CDLDOJI_Bull": 104, "CDLDOJI_Bear": 104
}

# Simplified descriptions for common patterns
PATTERN_DESCRIPTIONS = {
    "ENGULFING": "A reversal pattern where a candle body completely overshadows the previous one.",
    "HAMMER": "A bullish reversal pattern. Small body near the top, with a long lower shadow.",
    "HANGINGMAN": "A bearish reversal pattern. Small body near the top, with a long lower shadow.",
    "DOJI": "Indicates indecision. Opening and closing prices are virtually the same.",
    "MORNINGSTAR": "A bullish 3-candle reversal pattern: Long red, small body, long green.",
    "EVENINGSTAR": "A bearish 3-candle reversal pattern: Long green, small body, long red.",
    "MARUBOZU": "Strong trend signal. A candle with no shadows (wicks), just a full body.",
    "SHOOTINGSTAR": "A bearish reversal pattern. Small body near the bottom, with a long upper shadow."
}

# --- Functions ---

@st.cache_data(ttl=60) # Cache for 1 minute
def fetch_data(symbol, interval):
    """
    Fetch data from Yahoo Finance.
    Intervals: 1d, 1wk, 1mo (Intraday like 1h requires recent data only)
    """
    try:
        # Yahoo Finance ticker format for crypto is often "BTC-USD"
        ticker = symbol.replace("USDT", "-USD") if "USDT" in symbol else symbol
        
        # Download data
        df = yf.download(ticker, period="1y", interval=interval, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        # Clean up MultiIndex columns if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # Rename date/datetime to timestamp
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'timestamp'}, inplace=True)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Reorder/Ensure columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols].copy()
        
        # Convert to float
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def detect_patterns(df):
    candle_names = talib.get_function_groups()['Pattern Recognition']
    
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']

    for candle in candle_names:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)
        
    df['candlestick_pattern'] = "NO_PATTERN"
    df['candlestick_match_count'] = 0
    
    # We will prioritize valid patterns
    for index, row in df.iterrows():
        # Find patterns for this row
        patterns_found = []
        for candle in candle_names:
            if row[candle] != 0:
                patterns_found.append((candle, row[candle]))
        
        if patterns_found:
            # Rank patterns
            ranked_patterns = []
            for name, value in patterns_found:
                direction = "Bull" if value > 0 else "Bear"
                full_name = f"{name}_{direction}"
                rank = PATTERN_RANKINGS.get(full_name, 999)
                ranked_patterns.append((full_name, rank, name)) # Keep base name
            
            # Sort by rank
            ranked_patterns.sort(key=lambda x: x[1])
            best_pattern = ranked_patterns[0][0]
            
            df.loc[index, 'candlestick_pattern'] = best_pattern
            df.loc[index, 'candlestick_match_count'] = len(ranked_patterns)
    
    # Helper to get base name without Bull/Bear/CDL
    df['pattern_display'] = df['candlestick_pattern'].apply(
        lambda x: x.replace('NO_PATTERN', '').replace('CDL', '').replace('_Bull', '').replace('_Bear', '')
    )
    
    return df

# --- UI Layout ---
st.title("🕯️ Crypto Candlestick Pattern Watcher")
st.markdown("Real-time data via **Yahoo Finance**. Detects 50+ patterns.")

# Sidebar
st.sidebar.header("Configuration")
symbol_input = st.sidebar.text_input("Symbol", value="BTC-USD").upper()
interval_map = {"1 Day": "1d", "1 Week": "1wk"}
interval_selection = st.sidebar.selectbox("Timeframe", list(interval_map.keys()), index=0)

if st.sidebar.button("Scan Patterns"):
    with st.spinner(f"Fetching {symbol_input}..."):
        df = fetch_data(symbol_input, interval_map[interval_selection])
        
    if not df.empty:
        df = detect_patterns(df)
        last_row = df.iloc[-1]
        
        # --- Top Section: Metric ---
        st.metric(
            label=f"{symbol_input} Price", 
            value=f"${last_row['close']:.2f}",
            delta=f"{((last_row['close'] - df.iloc[-2]['close'])/df.iloc[-2]['close'])*100:.2f}%"
        )
        
        # --- Main Chart ---
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name=symbol_input
        )])
        
        # Highlight patterns
        patterns = df[df['candlestick_pattern'] != 'NO_PATTERN']
        if not patterns.empty:
            # Add markers for patterns
            fig.add_trace(go.Scatter(
                x=patterns['timestamp'],
                y=patterns['high'] * 1.02,
                mode='markers',
                marker=dict(symbol='arrow-down', size=10, color='orange'),
                name='Pattern Detected',
                hovertext=patterns['candlestick_pattern']
            ))

        fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{symbol_input} Chart")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Pattern Analysis Section ---
        st.subheader("🔍 Pattern Details & Comparison")
        
        # Get latest pattern
        latest_pattern_row = patterns.iloc[-1] if not patterns.empty else None
        
        if latest_pattern_row is not None:
            pattern_name = latest_pattern_row['candlestick_pattern']
            base_name = latest_pattern_row['pattern_display']
            timestamp = latest_pattern_row['timestamp']
            
            st.info(f"Most Recent Pattern: **{pattern_name}** detected on {timestamp.strftime('%Y-%m-%d')}")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 📸 Current Chart Zoom")
                # Create a mini chart zoomed into that specific day (+/- 5 candles)
                idx = df[df['timestamp'] == timestamp].index[0]
                start_idx = max(0, idx - 5)
                end_idx = min(len(df), idx + 5)
                zoom_df = df.iloc[start_idx:end_idx]
                
                zoom_fig = go.Figure(data=[go.Candlestick(
                    x=zoom_df['timestamp'],
                    open=zoom_df['open'], high=zoom_df['high'],
                    low=zoom_df['low'], close=zoom_df['close']
                )])
                zoom_fig.update_layout(height=300, xaxis_rangeslider_visible=False, title="Your Pattern Context")
                # Highlight the specific candle
                zoom_fig.add_shape(type="rect",
                    x0=timestamp - pd.Timedelta(hours=12) if 'd' in interval_map[interval_selection] else timestamp - pd.Timedelta(days=3),
                    x1=timestamp + pd.Timedelta(hours=12) if 'd' in interval_map[interval_selection] else timestamp + pd.Timedelta(days=3),
                    y0=zoom_df['low'].min(), y1=zoom_df['high'].max(),
                    line=dict(color="RoyalBlue", width=2),
                    fillcolor="LightSkyBlue", opacity=0.3
                )
                st.plotly_chart(zoom_fig, use_container_width=True)
                
            with c2:
                st.markdown("### 📘 Pattern Reference")
                desc = PATTERN_DESCRIPTIONS.get(base_name, "No description available for this specific pattern.")
                st.warning(f"**Definition**: {desc}")
                st.markdown("""
                > **Note on Similarity**: TA-Lib checks for strict mathematical adherence to the pattern definition. 
                > If a pattern is displayed, it is a **100% match** according to the algorithm. 
                > There is no 'partial' match percentage.
                """)
                
                # Placeholder for logic if we had images
                st.text("Idealized Pattern Structure:")
                if "DOJI" in base_name:
                    st.code("   |   \n --+--  (Open ~= Close)\n   |   ")
                elif "HAMMER" in base_name:
                    st.code("   __   \n  |  |  (Small Body)\n  |__|  \n   |    \n   |    (Long Shadow)\n   |   ")
                else:
                    st.markdown("*Checking 50+ patterns... specific diagram not available.*")

        else:
            st.write("No patterns detected in the current timeframe.")
            
    else:
        st.error(f"Could not load data for {symbol_input}. Try a valid Yahoo Ticker (e.g., BTC-USD).")
