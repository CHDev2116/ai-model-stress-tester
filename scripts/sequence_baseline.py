#!/usr/bin/env python3
"""
Neural tabular baseline (MLP) vs Random Forest on return targets.
For LSTM, use scripts/lstm_baseline.py (pip install ".[torch]").
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.config import load_config  # noqa: E402
from pipeline.models import create_regressor  # noqa: E402
from pipeline.paths import resolve_path  # noqa: E402
from pipeline.training import TRAIN_FRACTION, load_training_frame, walk_forward_predictions  # noqa: E402


def evaluate_mlp(df, config):
    features = [f for f in config["features"] if f in df.columns]
    X, y = df[features], df["Target"]
    split = int(len(df) * TRAIN_FRACTION)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
    factory = lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        random_state=42,
    )
    preds = walk_forward_predictions(X_scaled, y, split, model_factory=factory)
    y_test = y.iloc[split:]
    mae = float(mean_absolute_error(y_test, preds))
    return {
        "model": "mlp",
        "mae": mae,
        "mae_pct": round(mae * 100, 4),
        "direction_accuracy": float(np.mean(np.sign(y_test.values) == np.sign(preds))),
    }


def evaluate_rf(df, config):
    features = [f for f in config["features"] if f in df.columns]
    X, y = df[features], df["Target"]
    split = int(len(df) * TRAIN_FRACTION)
    preds = walk_forward_predictions(
        X, y, split, model_factory=lambda: create_regressor("rf")
    )
    y_test = y.iloc[split:]
    mae = float(mean_absolute_error(y_test, preds))
    return {
        "model": "random_forest",
        "mae": mae,
        "mae_pct": round(mae * 100, 4),
        "direction_accuracy": float(np.mean(np.sign(y_test.values) == np.sign(preds))),
    }


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    config = load_config(ticker)
    df = load_training_frame(ticker, config)

    report = {
        "ticker": ticker,
        "note": "MLP tabular baseline; see scripts/lstm_baseline.py for LSTM.",
        "random_forest": evaluate_rf(df, config),
        "mlp": evaluate_mlp(df, config),
    }
    print(json.dumps(report, indent=2))

    out = resolve_path(
        f"reports/sequence_baseline_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"📄 Saved: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
