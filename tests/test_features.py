import pandas as pd

from pipeline.features import apply_lookback, engineer_dataframe


def test_engineer_log_return_target(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    assert "Target" in df.columns
    assert "Volatility_20" in df.columns
    assert len(df) >= 200
    assert df["Target"].notna().all()


def test_engineer_price_target_override(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config, target_type="Price")
    assert df["Target"].iloc[-1] != df["Close"].iloc[-1]


def test_apply_lookback_mu_window(synthetic_ohlcv, goog_config):
    config = {**goog_config, "lookback_days": 60}
    df = engineer_dataframe(synthetic_ohlcv, config)
    before = len(df)
    after = len(apply_lookback(df, config))
    assert after < before
