import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from sklearn.preprocessing import StandardScaler

from pipeline.features import engineer_dataframe

ROOT = Path(__file__).resolve().parents[1]


def _load_lstm_script():
    path = ROOT / "scripts" / "lstm_baseline.py"
    spec = importlib.util.spec_from_file_location("lstm_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lstm_baseline"] = module
    spec.loader.exec_module(module)
    return module


def test_build_sequences_shape(synthetic_ohlcv, goog_config):
    lstm = _load_lstm_script()
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    features = [f for f in goog_config["features"] if f in df.columns]
    X, y = df[features], df["Target"]
    X_scaled = StandardScaler().fit_transform(X)
    seq_X, seq_y, indices = lstm.build_sequences(X_scaled, y, lookback=lstm.LOOKBACK)
    assert seq_X.shape[0] == len(indices)
    assert seq_X.shape[1] == lstm.LOOKBACK
    assert seq_X.shape[2] == len(features)
    assert len(seq_y) == len(indices)


@pytest.mark.slow
def test_evaluate_lstm_runs(synthetic_ohlcv, goog_config):
    lstm = _load_lstm_script()
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    metrics = lstm.evaluate_lstm(df, goog_config)
    assert metrics["model"] == "lstm"
    assert metrics["mae"] >= 0
    assert metrics["n_test_seq"] >= 10
