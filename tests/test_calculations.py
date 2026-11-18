import itertools
import math

import numpy as np
import pandas as pd
import pytest


# --- Helpers ---

def test_safe_pct_change(appmod):
    assert appmod.safe_pct_change(110, 100) == pytest.approx(10.0)
    assert appmod.safe_pct_change(90, 100) == pytest.approx(-10.0)
    for bad in (0, np.nan, np.inf):
        assert appmod.safe_pct_change(100, bad) == 0.0
    assert appmod.safe_pct_change(np.nan, 100) == 0.0


def test_safe_series_diff_matches_pandas_diff(appmod):
    s = pd.Series(np.random.default_rng(1).normal(size=60).cumsum())
    for lookback in (1, 10, 52):
        assert appmod.safe_series_diff(s, lookback) == pytest.approx(s.diff(lookback).iloc[-1])
    assert appmod.safe_series_diff(s.head(10), 10) == 0.0  # needs lookback + 1 bars


@pytest.mark.parametrize("value, expected", [
    (64658.328, "$64,658.33"),
    (1.5, "$1.50"),
    (0.31, "$0.3100"),
    (0.0000123456, "$0.00001235"),
    (0, "N/A"),
    (-3.3, "N/A"),
    (None, "N/A"),
    (float("nan"), "N/A"),
])
def test_format_price(appmod, value, expected):
    assert appmod.format_price(value) == expected


def test_format_pct(appmod):
    assert appmod.format_pct(12.345) == "+12.3%"
    assert appmod.format_pct(-0.04) == "-0.0%"
    assert appmod.format_pct(None) == "N/A"


# --- Symbols ---

@pytest.mark.parametrize("symbol, valid", [
    ("BTC-USD", True), ("btc-usd", True), ("BTCUSDT", True), ("ETHUSD", True),
    ("1INCH-USD", True), ("UNI7083-USD", True),
    ("BTC-EUR", False), ("BTC", False), ("123-USD", False), ("BTC-USD-X", False), ("", False),
])
def test_validate_symbol(appmod, symbol, valid):
    assert appmod.validate_symbol(symbol)[0] is valid


@pytest.mark.parametrize("symbol, expected", [
    ("BTC-USD", "BTCUSDT_PERP.A"),
    ("BTCUSDT", "BTCUSDT_PERP.A"),
    ("ETHUSD", "ETHUSDT_PERP.A"),
    ("USDCUSD", "USDCUSDT_PERP.A"),
    # A dashed base must not lose a "USD" suffix: TUSD is not the token T
    ("TUSD-USD", "TUSDUSDT_PERP.A"),
    ("FDUSD-USD", "FDUSDUSDT_PERP.A"),
])
def test_convert_symbol_to_coinalyze(appmod, symbol, expected):
    assert appmod.convert_symbol_to_coinalyze(symbol) == expected


# --- A/D phase, Wyckoff, signals ---

def test_phase_uptrend_when_ad_and_price_rise(appmod, ohlcv):
    df = ohlcv(np.linspace(100, 200, 80), clv=0.8)
    phase, color, df = appmod.analyze_ad_phase_fast(df, 20)
    assert (phase, color) == ("uptrend", "green")


def test_phase_accumulation_when_price_falls_but_ad_rises(appmod, ohlcv):
    # Closes near the bar highs make A/D rise while price falls: bullish divergence
    df = ohlcv(np.linspace(200, 100, 80), clv=0.8)
    phase, _, _ = appmod.analyze_ad_phase_fast(df, 20)
    assert phase == "accumulation"


def test_phase_distribution_when_price_rises_but_ad_falls(appmod, ohlcv):
    df = ohlcv(np.linspace(100, 200, 80), clv=-0.8)
    phase, _, _ = appmod.analyze_ad_phase_fast(df, 20)
    assert phase == "distribution"


def test_phase_neutral_without_enough_bars(appmod, ohlcv):
    phase, _, _ = appmod.analyze_ad_phase_fast(ohlcv(np.linspace(100, 200, 20), clv=0.8), 20)
    assert phase == "neutral"


def test_wyckoff_requires_lookback_plus_one_bars(appmod, ohlcv):
    _, _, df = appmod.analyze_ad_phase_fast(ohlcv(np.linspace(100, 200, 52), clv=0.8), 52)
    assert appmod.detect_wyckoff_enhanced(df, 52)["label"] == "N/A"
    _, _, df = appmod.analyze_ad_phase_fast(ohlcv(np.linspace(100, 200, 53), clv=0.8), 52)
    assert appmod.detect_wyckoff_enhanced(df, 52)["label"] == "UPTREND"


def test_generate_signals(appmod):
    df = pd.DataFrame({
        "candlestick_pattern": ["CDLHAMMER_Bull", "CDLENGULFING_Bear", "NO_PATTERN", "CDLHAMMER_Bull"],
        "pattern_direction": ["bullish", "bearish", "neutral", "bullish"],
        "phase": ["accumulation", "distribution", "accumulation", "downtrend"],
    })
    out = appmod.generate_signals_fast(df)
    assert out["signal"].tolist() == ["strong_buy", "strong_sell", "none", "none"]


def test_detect_patterns_only_reports_ranked_patterns(appmod):
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400)))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=400, freq="D"),
        "open": close * (1 + rng.normal(0, 0.01, 400)),
        "high": close * 1.03, "low": close * 0.97, "close": close, "volume": 1.0,
    })
    out = appmod.detect_patterns_optimized(df)
    found = out.loc[out["candlestick_pattern"] != "NO_PATTERN", "candlestick_pattern"]
    assert not found.empty
    assert found.isin(appmod.PATTERN_RANKINGS.keys()).all()
    assert (out.loc[found.index, "pattern_direction"] != "neutral").all()
    assert (out.loc[out["candlestick_pattern"] == "NO_PATTERN", "candlestick_match_count"] == 0).all()


def test_phase_zones_cover_each_bar_fully(appmod):
    ts = pd.date_range("2024-01-01", periods=7, freq="D")
    df = pd.DataFrame({"timestamp": ts, "phase": [
        "neutral", "accumulation", "accumulation", "neutral", "distribution", "neutral", "accumulation"]})
    zones = appmod.get_phase_zones_fast(df)
    assert zones == [
        {"phase": "accumulation", "start": ts[1], "end": ts[3]},
        {"phase": "distribution", "start": ts[4], "end": ts[5]},  # one-bar zone is not zero-width
        {"phase": "accumulation", "start": ts[6], "end": ts[6] + pd.Timedelta(days=1)},
    ]


# --- Scores ---

def test_signal_strength_does_not_count_phase_twice(appmod, ohlcv):
    # Accumulation phase, but the A/D line has just dropped below its EMA:
    # the A/D factor must follow the EMA, not repeat the phase.
    _, _, df = appmod.analyze_ad_phase_fast(ohlcv(np.linspace(200, 100, 80), clv=0.8), 20)
    df.loc[df.index[-1], "ad"] = df["ad_ema"].iloc[-1] - 1
    _, breakdown = appmod.calculate_signal_strength("accumulation", {"label": "N/A"}, df, 20)
    assert breakdown["phase"] == 25
    assert breakdown["ad_momentum"] == -10


def test_signal_strength_score_is_clamped_sum_of_breakdown(appmod, ohlcv):
    for closes, clv in [(np.linspace(100, 200, 80), 0.8), (np.linspace(200, 100, 80), -0.8)]:
        phase, _, df = appmod.analyze_ad_phase_fast(ohlcv(closes, clv=clv), 20)
        wyckoff = appmod.detect_wyckoff_enhanced(df, 20)
        score, breakdown = appmod.calculate_signal_strength(phase, wyckoff, df, 20)
        assert score == max(0, min(100, 50 + sum(breakdown.values())))


@pytest.mark.parametrize("oi, price, expected_points", [
    (5, 5, 20),     # new longs
    (5, -5, -20),   # short buildup: bearish per the guide, must lower the score
    (-5, 5, 0),     # short covering
    (-5, -5, -5),   # capitulation
    (0.5, 5, 0),    # OI flat
    (5, 0.5, 0),    # price flat
])
def test_derivatives_oi_price_regime(appmod, oi, price, expected_points):
    result = appmod.calculate_derivatives_score(oi, funding_rate=0.01, long_short_ratio=1.0, price_change=price)
    assert result["score"] == 50 + expected_points


@pytest.mark.parametrize("funding, expected_points", [
    (-0.02, 15), (-0.01, 5), (-0.001, 5), (0, 0), (0.01, 0), (0.02, -5), (0.05, -5), (0.06, -15),
])
def test_derivatives_funding_bands(appmod, funding, expected_points):
    result = appmod.calculate_derivatives_score(0, funding, long_short_ratio=1.0, price_change=0)
    assert result["score"] == 50 + expected_points


@pytest.mark.parametrize("ratio, expected_points", [
    (0.7, 15), (0.8, 5), (0.95, 5), (1.0, 0), (1.2, 0), (1.3, -5), (1.5, -5), (1.6, -15),
])
def test_derivatives_long_short_bands(appmod, ratio, expected_points):
    result = appmod.calculate_derivatives_score(0, 0, long_short_ratio=ratio, price_change=0)
    assert result["score"] == 50 + expected_points


def test_derivatives_score_spans_0_to_100(appmod):
    scores = [
        appmod.calculate_derivatives_score(oi, fr, ls, px)["score"]
        for oi, fr, ls, px in itertools.product(
            [-5, -0.5, 0, 0.5, 5], [-0.05, -0.005, 0, 0.01, 0.03, 0.1], [0.5, 0.9, 1.0, 1.3, 2.0], [-5, 0, 5])
    ]
    assert min(scores) == 0 and max(scores) == 100


def test_derivatives_factors_explain_score(appmod):
    result = appmod.calculate_derivatives_score(5, -0.02, 0.7, 5)
    assert result["score"] == min(100, 50 + sum(int(points) for _, points, _ in result["factors"]))


def test_adx_direction_comes_from_di_not_strength_label(appmod, ohlcv):
    up = appmod.calculate_trend_strength_adx(ohlcv(np.linspace(100, 200, 120)))
    down = appmod.calculate_trend_strength_adx(ohlcv(np.linspace(200, 100, 120)))
    assert up["direction"] == "bullish" and down["direction"] == "bearish"
    assert up["strength"] == down["strength"]


def test_mtf_alignment_bounds_and_flat_trend(appmod, ohlcv):
    up_d = ohlcv(np.linspace(100, 200, 300), clv=0.8)
    up_w = ohlcv(np.linspace(100, 200, 120), freq="W-MON", clv=0.8)
    assert appmod.calculate_mtf_alignment(up_d, up_w)["score"] == 100
    down_d = ohlcv(np.linspace(200, 100, 300), clv=-0.8)
    down_w = ohlcv(np.linspace(200, 100, 120), freq="W-MON", clv=-0.8)
    assert appmod.calculate_mtf_alignment(down_d, down_w)["score"] == 0
    flat = appmod.calculate_mtf_alignment(ohlcv(np.full(300, 100.0)), ohlcv(np.full(120, 100.0), freq="W-MON"))
    assert flat["factors"]["trend_alignment"] == 0


# --- Levels ---

def test_support_resistance_pivots(appmod, ohlcv):
    df = ohlcv(np.linspace(100, 120, 60))
    levels = appmod.find_support_resistance(df, 52)
    recent = df.tail(52)
    high, low, close = recent["high"].max(), recent["low"].min(), recent["close"].iloc[-1]
    pivot = (high + low + close) / 3
    assert levels["pivot"] == pytest.approx(pivot)
    assert levels["support"] == pytest.approx(2 * pivot - high)
    assert levels["resistance"] == pytest.approx(2 * pivot - low)
    assert levels["resistance"] - levels["support"] == pytest.approx(high - low)
    assert levels["entry_low"] <= levels["entry_high"] <= levels["pivot"]


def test_support_unavailable_after_deep_drawdown(appmod):
    close = np.linspace(100, 25, 60)
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
                       "open": close, "high": close * 1.01, "low": close * 0.8, "close": close, "volume": 1.0})
    levels = appmod.find_support_resistance(df, 52)
    assert levels["support"] == 0 and levels["entry_low"] == 0 and levels["entry_high"] == 0
    assert levels["resistance"] > 0


# --- Performance ---

def test_period_returns_are_close_to_close(appmod, ohlcv):
    df = ohlcv(np.arange(1, 401, dtype=float), start="2025-01-01")
    perf = appmod.calculate_performance_metrics(df)
    last = df["close"].iloc[-1]
    assert perf["7d"] == pytest.approx((last / df["close"].iloc[-8] - 1) * 100)
    assert perf["30d"] == pytest.approx((last / df["close"].iloc[-31] - 1) * 100)
    assert perf["90d"] == pytest.approx((last / df["close"].iloc[-91] - 1) * 100)


def test_ytd_uses_prior_year_last_close(appmod, ohlcv):
    df = ohlcv(np.arange(1, 401, dtype=float), start="2025-01-01")
    prior_close = df.loc[df["timestamp"].dt.year == 2025, "close"].iloc[-1]
    perf = appmod.calculate_performance_metrics(df)
    assert perf["ytd"] == pytest.approx((df["close"].iloc[-1] / prior_close - 1) * 100)


def test_returns_longer_than_history_are_unavailable(appmod, ohlcv):
    df = ohlcv(np.linspace(10, 20, 20), start="2026-10-01")
    perf = appmod.calculate_performance_metrics(df)
    assert perf["7d"] is not None
    assert perf["30d"] is None and perf["90d"] is None
    assert any("30D return unavailable" in w for w in perf["warnings"])


def test_last_bar_is_forming(appmod):
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    assert appmod.last_bar_is_forming(pd.DataFrame({"timestamp": [today]}), "1d")
    assert not appmod.last_bar_is_forming(pd.DataFrame({"timestamp": [today - pd.Timedelta(days=3)]}), "1d")


# --- Data source parsing (no network) ---

def test_cryptocompare_weekly_bars_are_monday_to_sunday(appmod, monkeypatch):
    days = pd.date_range("2024-01-01", "2024-01-21", freq="D")  # starts on a Monday
    rows = [{"time": int(d.timestamp()), "open": i + 1.0, "high": i + 2.0, "low": i + 0.5,
             "close": i + 1.5, "volumeto": 10.0} for i, d in enumerate(days)]

    class FakeResponse:
        def json(self):
            return {"Response": "Success", "Data": rows}

    monkeypatch.setattr(appmod, "get_cryptocompare_api_key", lambda: "test-key")
    monkeypatch.setattr(appmod.requests, "get", lambda *a, **k: FakeResponse())
    weekly = appmod.fetch_data_cryptocompare_raw("BTC-USD", "1wk")
    assert len(weekly) == 3
    assert (weekly["timestamp"].dt.dayofweek == 0).all()
    first = weekly.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (1.0, 8.0, 0.5, 7.5, 70.0)


# --- Caching ---

def test_cache_unless_failed_retries_failures_and_keeps_functions_apart(appmod):
    calls = {"a": 0, "b": 0}

    @appmod.cache_unless_failed(lambda r: r is None)
    def flaky(x):
        calls["a"] += 1
        return None if calls["a"] == 1 else x * 2

    @appmod.cache_unless_failed(lambda r: r is None)
    def other(x):
        calls["b"] += 1
        return x * 100

    assert flaky(3) is None          # failure returned...
    assert flaky(3) == 6             # ...but not cached, so retried
    assert flaky(3) == 6 and calls["a"] == 2  # success is cached
    assert other(3) == 300           # same args, separate cache
    flaky.clear()
    other.clear()
