
import streamlit as st
import pandas as pd
import numpy as np
import talib
import requests
import plotly.graph_objects as go
from itertools import compress

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Pattern Watcher")

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

# --- Functions ---

@st.cache_data(ttl=60) # Cache for 1 minute
def fetch_binance_data(symbol, interval):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": 500}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching data from Binance: {e}")
        return pd.DataFrame()

def detect_patterns(df):
    candle_names = talib.get_function_groups()['Pattern Recognition']
    # Filter out removed patterns as per original notebook logic or keep all
    # Using all for now or filtering based on notebook preferences if needed
    
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']

    for candle in candle_names:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)
        
    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan

    # Identify patterns (This part can be heavy on rows loop, but manageable for 500 rows)
    # We will just iterate over the last few rows for performance if needed, but 500 is fast.
    
    for index, row in df.iterrows():
        # Get pattern columns that are not 0
        patterns_found = []
        for candle in candle_names:
            if row[candle] != 0:
                patterns_found.append((candle, row[candle]))
        
        if not patterns_found:
            df.loc[index, 'candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, 'candlestick_match_count'] = 0
        else:
            # Determine Bull/Bear and Rank
            ranked_patterns = []
            for name, value in patterns_found:
                direction = "Bull" if value > 0 else "Bear"
                full_name = f"{name}_{direction}"
                rank = PATTERN_RANKINGS.get(full_name, 999) # Default high rank if not found
                ranked_patterns.append((full_name, rank))
            
            # Sort by rank (lower is better)
            ranked_patterns.sort(key=lambda x: x[1])
            best_pattern = ranked_patterns[0][0]
            
            df.loc[index, 'candlestick_pattern'] = best_pattern
            df.loc[index, 'candlestick_match_count'] = len(ranked_patterns)
    
    # Clean up pattern name
    df['candlestick_pattern'] = df['candlestick_pattern'].fillna('NO_PATTERN')
    # Remove 'CDL' prefix for display
    df['pattern_display'] = df['candlestick_pattern'].apply(lambda x: x.replace('NO_PATTERN', '').replace('CDL', ''))
    
    return df

# --- UI Layout ---
st.title("🕯️ Crypto Candlestick Pattern Watcher")
st.markdown("Retrieves real-time data from **Binance** and detects over 50+ candlestick patterns.")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Symbol", value="BTCUSDT").upper()
interval_map = {
    "1 Day": "1d",
    "1 Week": "1w",
    "1 Hour": "1h",
    "4 Hours": "4h"
}
interval_selection = st.sidebar.selectbox("Timeframe", list(interval_map.keys()), index=0)
interval = interval_map[interval_selection]

if st.sidebar.button("Scan Patterns"):
    with st.spinner(f"Fetching {symbol} ({interval}) data..."):
        df = fetch_binance_data(symbol, interval)
        
    if not df.empty:
        # Detect
        df = detect_patterns(df)
        
        # Display Current Status
        last_row = df.iloc[-1]
        st.metric(
            label=f"{symbol} Current Price", 
            value=f"${last_row['close']:.2f}",
            delta=f"{((last_row['close'] - df.iloc[-2]['close'])/df.iloc[-2]['close'])*100:.2f}%"
        )

        # Plotly Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )])
        
        # Add markers for detected patterns (optional/advanced: add annotations)
        # For now, let's highlight where patterns are found
        patterns_detected = df[df['candlestick_pattern'] != 'NO_PATTERN']
        
        fig.update_layout(
            title=f"{symbol} Candlestick Chart ({interval})",
            yaxis_title="Price (USDT)",
            xaxis_title="Time",
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Patterns Table
        st.subheader("Recent Detected Patterns")
        
        # Filter mostly interesting columns
        display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'pattern_display']
        # Show last 10 patterns found
        recent_patterns = patterns_detected.tail(10).sort_values(by='timestamp', ascending=False)
        
        if not recent_patterns.empty:
            st.dataframe(recent_patterns[display_cols].style.applymap(
                lambda x: 'color: green' if 'Bull' in str(x) else ('color: red' if 'Bear' in str(x) else ''),
                subset=['pattern_display']
            ))
        else:
            st.info("No significant patterns detected in the recent data.")

        # Show raw data option
        with st.expander("View Raw Data"):
            st.dataframe(df)

else:
    st.info("Enter a symbol and click 'Scan Patterns' to start.")
