#!/usr/bin/env python3
"""Benchmark Random Forest vs XGBoost vs LightGBM (optional deps)."""
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.metrics import mean_absolute_error

from pipeline.config import load_all_configs, load_config  # noqa: E402
from pipeline.models import create_regressor  # noqa: E402
from pipeline.paths import resolve_path  # noqa: E402
from pipeline.training import TRAIN_FRACTION, load_training_frame, predict_targets, prepare_xy  # noqa: E402


def evaluate_model(df, config, model_name):
    target_type = config["target_type"]
    X, y, features = prepare_xy(df, config)
    split = int(len(df) * TRAIN_FRACTION)
    factory = lambda m=model_name: create_regressor(m)  # noqa: E731
    preds = predict_targets(X, y, split, target_type, model_factory=factory)
    y_test = y.iloc[split:]
    mae = float(mean_absolute_error(y_test, preds))
    result = {
        "model": model_name,
        "mae": mae,
        "features": features,
    }
    if "Return" in target_type:
        result["mae_pct"] = round(mae * 100, 4)
        import numpy as np

        result["direction_accuracy"] = float(
            np.mean(np.sign(y_test.values) == np.sign(preds))
        )
    return result


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or list(load_all_configs().keys())
    models = ["random_forest", "xgboost", "lightgbm"]
    report = {"timestamp": datetime.now().isoformat(), "tickers": {}}

    for ticker in tickers:
        config = load_config(ticker)
        df = load_training_frame(ticker, config)
        report["tickers"][ticker] = {}
        for model_name in models:
            try:
                metrics = evaluate_model(df, config, model_name)
                report["tickers"][ticker][model_name] = metrics
                label = metrics.get("mae_pct", metrics["mae"])
                print(f"{ticker} {model_name}: {label}")
            except ImportError as exc:
                report["tickers"][ticker][model_name] = {"error": str(exc)}
                print(f"{ticker} {model_name}: skipped ({exc})")

    out = resolve_path(
        f"reports/model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"📄 Saved: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
