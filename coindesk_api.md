# CoinDesk Cryptocurrency Data API - Quick Reference

**Base URL:** `https://min-api.cryptocompare.com`  
**Cache:** Mostly 10 seconds (varies by endpoint)  
**Auth:** Requires API key

---

## 1. Current Price Endpoints

### Single Crypto → Multiple Targets
```
GET /data/price
?fsym=BTC           # Required: From symbol (e.g., BTC, ETH)
&tsyms=USD,EUR      # Required: To symbols (comma-separated)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&tryConversion=true # Optional: Allow indirect conversions via BTC/ETH (default: true)
&relaxedValidation=true # Optional: Skip non-trading pairs (default: true)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Multiple Cryptos → Multiple Targets
```
GET /data/pricemulti
?fsyms=BTC,ETH,SOL  # Required: From symbols (comma-separated, max 300)
&tsyms=USD,EUR      # Required: To symbols (comma-separated, max 100)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&relaxedValidation=true # Optional: Skip non-trading pairs (default: true)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Multiple Cryptos Full Data (Price + Vol + OHLC)
```
GET /data/pricemultifull
?fsyms=BTC,ETH      # Required: From symbols (comma-separated, max 1000)
&tsyms=USD,EUR      # Required: To symbols (comma-separated, max 100)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&relaxedValidation=true # Optional: Skip non-trading pairs (default: true)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Custom Average (Volume-Weighted)
```
GET /data/generateAvg
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol
&e=KRAKEN,BITSTAMP  # Required: Exchange list (comma-separated, max 150)
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

---

## 2. Historical Data Endpoints

### Daily OHLCV
```
GET /data/histoday
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol
&limit=30           # Optional: Data points to return (default: 30, max: 2000 with aggregation)
&aggregate=1        # Optional: Days to aggregate (default: 1)
&toTs=1609459200    # Optional: Unix timestamp end point (for pagination)
&allData=false      # Optional: Get ALL historical data (only daily)
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&aggregatePredictableTimePeriods=true # Optional: Fixed time slots (default: true)
&explainPath=false  # Optional: Show conversion path (default: false)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Hourly OHLCV
```
GET /data/histohour
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol
&limit=168          # Optional: Data points to return (default: 168)
&aggregate=1        # Optional: Hours to aggregate (default: 1)
&toTs=1609459200    # Optional: Unix timestamp end point
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&aggregatePredictableTimePeriods=true # Optional: Fixed time slots (default: true)
&explainPath=false  # Optional: Show conversion path (default: false)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Minute OHLCV (7 days only)
```
GET /data/histominute
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol
&limit=1440         # Optional: Data points to return (default: 1440)
&aggregate=1        # Optional: Minutes to aggregate (default: 1)
&toTs=1609459200    # Optional: Unix timestamp end point
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&aggregatePredictableTimePeriods=true # Optional: Fixed time slots (default: true)
&explainPath=false  # Optional: Show conversion path (default: false)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Daily Market Close CSV (Enterprise Only)
```
GET /data/daily/market/close
?date=2023-12-25    # Required: Date in YYYY-MM-DD format
```

### Minute OHLCV CSV by Date
```
GET /data/histominute
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol
&date=2023-12-25    # Required: Date in YYYY-MM-DD format
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
```

### Historical Price at Specific Timestamp
```
GET /data/pricehistorical
?fsym=BTC           # Required: From symbol
&tsyms=USD,EUR      # Required: To symbols (comma-separated)
&ts=1609459200      # Required: Unix timestamp
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&calculationType=Close # Optional: Close | MidHighLow | VolFVolT (default: Close)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Day Average Price (Hourly VWAP)
```
GET /data/dayavg
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol
&toTs=1609459200    # Optional: Unix timestamp (defaults to today)
&tryConversion=true # Optional: Allow indirect conversions (default: true)
&avgType=HourVWAP   # Optional: HourVWAP | MidHighLow | VolFVolT (default: HourVWAP)
&UTCHourDiff=0      # Optional: Timezone offset, e.g., -8 for PST (default: 0 UTC)
&e=CCCAGG           # Optional: Exchange (default: CCCAGG)
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Daily Symbol Volume
```
GET /data/histoday
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol (volume in this pair)
&limit=30           # Optional: Data points (default: 30)
&aggregate=1        # Optional: Days to aggregate (default: 1)
&toTs=1609459200    # Optional: Unix timestamp end point
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

### Hourly Symbol Volume
```
GET /data/histohour
?fsym=BTC           # Required: From symbol
&tsym=USD           # Required: To symbol (volume in this pair)
&limit=30           # Optional: Data points (default: 30)
&aggregate=1        # Optional: Hours to aggregate (default: 1)
&toTs=1609459200    # Optional: Unix timestamp end point
&extraParams=MyApp  # Optional: Your app name
&sign=false         # Optional: Sign for smart contracts (default: false)
```

---

## Key Parameter Limits & Defaults

| Parameter | Type | Min | Max | Default | Notes |
|-----------|------|-----|-----|---------|-------|
| fsym | string | 1 | 30 | — | Single crypto symbol |
| fsyms | string | 1 | 300–1000 | — | Comma-separated (varies by endpoint) |
| tsyms | string | 1 | 30–500 | — | Comma-separated |
| e | string | 2 | 30–150 | CCCAGG | Exchange name or CCCAGG aggregate |
| limit | int | 1 | 2000 | Varies | Max returned points (endpoint-dependent) |
| aggregate | int | 1 | ∞ | 1 | Period multiplier |
| toTs | timestamp | — | — | Now | Unix timestamp (seconds) |
| extraParams | string | 1 | 2000 | NotAvailable | App identifier |
| tryConversion | bool | — | — | true | Allow BTC/ETH intermediary conversion |
| relaxedValidation | bool | — | — | true | Skip non-trading pairs |
| sign | bool | — | — | false | Sign for smart contracts |

---

## Common Patterns for Code

### Pagination for Historical Data (Get All)
```
limit=2000&toTs={current_timestamp}
# Then loop:
toTs={earliest_timestamp_from_previous_response}
```

### Multi-Crypto Price Check
```
GET /data/pricemulti?fsyms=BTC,ETH,SOL,LINK&tsyms=USD,EUR&extraParams=MyBot
```

### Custom Weighted Average (Specific Exchanges)
```
GET /data/generateAvg?fsym=BTC&tsym=USD&e=KRAKEN,BITSTAMP,COINBASE&extraParams=MyBot
```

### Get Full Candle Data
```
GET /data/pricemultifull?fsyms=BTC,ETH&tsyms=USD&extraParams=MyBot
# Returns: price, volume, open, high, low, change, change% (+ display versions)
```

---

## Notes

- **Cache times:** Most endpoints cache for 10 seconds; daily data caches 610 seconds
- **Direct pairs only:** Set `tryConversion=false` to avoid BTC/ETH conversion fallback
- **CSV format:** Some endpoints support CSV (enterprise or legacy endpoints)
- **Timezone support:** Use `UTCHourDiff` param for day average calculations (e.g., `-8` for PST)
- **Volume conversions:** Historical volume data converts to the target currency at matched historical rates
