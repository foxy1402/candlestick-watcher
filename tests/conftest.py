import numpy as np
import pandas as pd
import pytest
import streamlit.logger

# Importing app.py runs the Streamlit script in "bare mode": widgets return their
# defaults (no button is pressed), so nothing is fetched. Silence the
# "missing ScriptRunContext" warnings that bare mode logs.
streamlit.logger.set_log_level("error")

import app  # noqa: E402


@pytest.fixture(scope="session")
def appmod():
    return app


def make_ohlcv(close, *, start="2024-01-01", freq="D", clv=0.0, volume=1_000_000.0):
    """OHLCV frame whose close sits at a fixed position in each bar's range.

    clv is the Close Location Value used by the A/D line: +1 closes at the high,
    -1 at the low, 0 mid-range. It makes the A/D direction deterministic.
    """
    close = np.asarray(close, dtype=float)
    spread = close * 0.02
    high = close + spread * (1 - clv) / 2
    low = close - spread * (1 + clv) / 2
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(close), freq=freq),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(len(close), volume),
    })


@pytest.fixture
def ohlcv():
    return make_ohlcv
