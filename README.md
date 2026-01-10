# 🕯️ Crypto Candlestick Pattern Watcher

A **Streamlit web application** that detects 50+ candlestick patterns and analyzes market phases (Accumulation/Distribution) using real-time data from Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)
![TA-Lib](https://img.shields.io/badge/TA--Lib-Pattern%20Recognition-green)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **50+ Pattern Detection** | Hammer, Doji, Engulfing, Morning Star, and more |
| **A/D Phase Analysis** | Detect Accumulation & Distribution divergences |
| **Interactive Charts** | Plotly candlestick charts with pattern markers |
| **Multi-Timeframe** | 1 Day & 1 Week candles for long-term analysis |

## 🚀 Live Demo

**[Open App](https://goodtrade.streamlit.app/)** *(Replace with your actual URL)*

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

## 📊 How It Works

### Pattern Detection
Uses TA-Lib's pattern recognition functions to scan candles and rank patterns by importance (e.g., Three Line Strike > Doji).

### Accumulation/Distribution
Calculates the **Chaikin A/D Line** and detects divergences:
- **Accumulation**: Price falling but A/D rising (smart money buying)
- **Distribution**: Price rising but A/D falling (smart money selling)

## 📝 License

MIT

## 🙏 Credits

- Original pattern logic inspired by [Caner Irfanoglu's article](https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5)
- [TA-Lib](https://github.com/mrjbq7/ta-lib)
