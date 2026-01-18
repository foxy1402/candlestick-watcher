# 🕯️ Crypto Candlestick Pattern Watcher

A **Streamlit web application** for long-term crypto investors. Analyzes market phases (Accumulation/Distribution), detects 50+ candlestick patterns, and provides entry signals using real-time data from Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)
![TA-Lib](https://img.shields.io/badge/TA--Lib-Pattern%20Recognition-green)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📋 Watchlist Dashboard** | Monitor multiple assets at a glance with A/D status |
| **🟢 Phase Zones** | Shaded chart areas showing Accumulation/Distribution periods |
| **🛒 Simplified Wyckoff** | Beginner-friendly labels: "Smart Money Buying/Selling" |
| **⭐ Entry Signals** | Strong Buy/Sell when A/D + Pattern align |
| **🔄 Timeframe Compare** | Daily vs Weekly alignment check |
| **50+ Patterns** | Hammer, Doji, Engulfing, Morning Star, and more |

## 🚀 Live Demo

**[Open App](https://goodtrade.streamlit.app/)**

## 📊 How to Use (For Long-Term Investors)

### 1. Watchlist Dashboard
Add your favorite coins (BTC, ETH, SOL, etc.) and see which ones are in **Accumulation** (🟢) or **Distribution** (🔴) at a glance.

### 2. Single Asset Analysis
Select a coin and click "Analyze" to see:
- **Phase Zones**: Green shaded areas = Accumulation periods
- **Wyckoff Phase**: Simple labels like "Smart Money Buying"
- **Entry Signals**: ⭐ markers when pattern + phase align

### 3. Timeframe Comparison
Check if Daily AND Weekly charts agree:
- ✅ Both bullish = High confidence entry
- ⚠️ Mixed = Wait for confirmation

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data**: Yahoo Finance (`yfinance`)
- **Analysis**: TA-Lib (Technical Analysis Library)
- **Charts**: Plotly

## 📦 Installation (Local)

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/candlestick-watcher.git
cd candlestick-watcher

# Install dependencies (requires TA-Lib C library)
pip install -r requirements.txt

# Run
streamlit run app.py
```

> ⚠️ **Note**: TA-Lib requires a C library. On Windows, use pre-built wheels. On Linux/Mac, install via `brew install ta-lib` or `apt-get install libta-lib-dev`.

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, select `app.py`
4. Deploy!

## 📝 License

MIT

## 🙏 Credits

- Original pattern logic inspired by [Caner Irfanoglu's article](https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5)
- [TA-Lib](https://github.com/mrjbq7/ta-lib)
