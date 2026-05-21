import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def goog_config():
    return {
        "data_source": "data/goog_assets.csv",
        "processed_output": "data/processed_goog.csv",
        "features": ["MA5", "MA20", "MA_Ratio", "Volatility_20", "RSI", "Bias"],
        "target_type": "Log_Return",
    }


@pytest.fixture
def synthetic_ohlcv():
    """~300 business days of synthetic prices for pipeline smoke tests."""
    n = 320
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0, 0.8, n))
    volume = rng.integers(800_000, 1_200_000, n)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "Volume": volume,
            "High": high,
            "Low": low,
        }
    )
