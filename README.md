# 🕯️ Crypto Candlestick Pattern Watcher

A **Streamlit web application** for long-term crypto investors. Analyzes market phases (Accumulation/Distribution), detects 50+ candlestick patterns, provides entry signals, and monitors derivatives data (Open Interest, Funding Rate, Long/Short Ratio).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)
![TA-Lib](https://img.shields.io/badge/TA--Lib-Pattern%20Recognition-green)
![Coinalyze](https://img.shields.io/badge/Coinalyze-API-orange)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📋 Watchlist Dashboard** | Monitor multiple assets at a glance with A/D status |
| **🟢 Phase Zones** | Shaded chart areas showing Accumulation/Distribution periods |
| **🛒 Simplified Wyckoff** | Beginner-friendly labels: "Smart Money Buying/Selling" |
| **⭐ Entry Signals** | Strong Buy/Sell when A/D + Pattern align |
| **🔄 Timeframe Compare** | Daily vs Weekly alignment check |
| **50+ Patterns** | Hammer, Doji, Engulfing, Morning Star, and more |
| **📈 Open Interest Monitor** | Derivatives analysis with investment scoring |

## 📈 Open Interest Monitor (NEW)

Analyze futures market data for smarter investment decisions:

- **Open Interest (OI)**: Track rising/falling OI with price divergence
- **Funding Rate**: Identify market sentiment (bullish/bearish bias)
- **Long/Short Ratio**: See market positioning at a glance
- **Investment Score**: 0-100 score combining all derivatives metrics
- **Historical Signals**: Visual markers showing past bullish/bearish signals

### Derivatives Signal Interpretation

| OI Change | Price Change | Signal | Meaning |
|-----------|-------------|--------|---------|
| ↑ Rising | ↑ Rising | 🟢 Bullish | New money entering, trend strengthening |
| ↑ Rising | ↓ Falling | 🔴 Bearish | Short sellers entering aggressively |
| ↓ Falling | ↑ Rising | 🟡 Weak Rally | Short covering, may not sustain |
| ↓ Falling | ↓ Falling | 🟠 Capitulation | Liquidations, potential bottom |

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

### 4. Open Interest Monitor
Select a coin and analyze derivatives data:
- **4-Hour Resolution**: ~3 months of data for short-term trends
- **Daily Resolution**: ~1 year of data for long-term analysis
- **Investment Score**: 0-100 rating based on derivatives health

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Sources**:
  - **Yahoo Finance** (`yfinance`) - Default, native daily + weekly support
  - **CryptoCompare** API - Alternative source, best for daily timeframe
- **Derivatives**: Coinalyze API
- **Analysis**: TA-Lib (Technical Analysis Library)
- **Charts**: Plotly

### Data Source Toggle

The app includes a toggle to switch between Yahoo Finance and CryptoCompare:
- **Yahoo Finance (Default)**: Recommended for weekly timeframe analysis as it provides native weekly candlestick data
- **CryptoCompare**: Great for daily timeframe, but weekly data is simulated by aggregating daily candles (less accurate for weekly analysis)

## 📦 Installation (Local)

```bash
# Clone repo
git clone https://github.com/foxy1402/candlestick-watcher.git
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
4. **Add Secrets** (Settings → Secrets):
   ```toml
   COINALYZE_API_KEY = "your-api-key-here"
   ```
5. Deploy!

### Getting a Coinalyze API Key
1. Sign up at [coinalyze.net](https://coinalyze.net)
2. Get your free API key (1000 calls/month)
3. Add to Streamlit Secrets as shown above

## 🐳 Docker / GHCR Deployment

Every push to `main`/`master` (and every `v*` tag) builds a container image and
publishes it to the GitHub Container Registry via GitHub Actions
(`.github/workflows/docker-publish.yml`). No registry secrets are required —
it uses the built-in `GITHUB_TOKEN`.

**Image:** `ghcr.io/<your-github-username>/<repo-name>:latest`

> After the first successful run, open the package on GitHub → **Package settings**
> and set visibility to **Public** if you want to pull it without authentication.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8501` | Port the app listens on. Most PaaS inject this automatically. |
| `COINALYZE_API_KEY` | – | Required for the Open Interest Monitor. |
| `CRYPTOCOMPARE_API_KEY` | – | Optional alternative price data source. |

### Run locally

```bash
docker run --rm -p 8501:8501 \
  -e COINALYZE_API_KEY=your-key \
  ghcr.io/<your-github-username>/<repo-name>:latest
# open http://localhost:8501
```

To run on a different port:

```bash
docker run --rm -e PORT=9000 -p 9000:9000 ghcr.io/<your-github-username>/<repo-name>:latest
```

### Deploy on Render

1. **New → Web Service → Deploy an existing image from a registry**
2. Image URL: `ghcr.io/<your-github-username>/<repo-name>:latest`
3. Render sets `PORT` automatically — the app binds to it, no extra config needed.
4. Add `COINALYZE_API_KEY` under **Environment**.

### Deploy on Portainer

1. **Stacks / Containers → Add container**
2. Image: `ghcr.io/<your-github-username>/<repo-name>:latest`
3. Map a host port to container port `8501` (or set `PORT` and map that port).
4. Add env vars (`PORT`, `COINALYZE_API_KEY`) under **Env**.

### Build the image yourself

```bash
docker build -t candlestick-watcher .
docker run --rm -p 8501:8501 candlestick-watcher
```

## 📝 License

MIT

## 🙏 Credits

- Original pattern logic inspired by [Caner Irfanoglu's article](https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5)
- [TA-Lib](https://github.com/mrjbq7/ta-lib)
- [Coinalyze API](https://coinalyze.net) for derivatives data

