"""Train, persist, and serve latest-tabular models."""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pipeline.config import load_config
from pipeline.features import apply_lookback, engineer_dataframe
from pipeline.paths import resolve_path
from pipeline.training import REGRESSOR_PARAMS, prepare_xy


def _artifacts_dir():
    path = resolve_path("artifacts")
    os.makedirs(path, exist_ok=True)
    return path


def load_training_frame(ticker, config=None, target_type_override=None):
    config = config or load_config(ticker)
    if target_type_override:
        raw_path = resolve_path(config["data_source"])
        df = pd.read_csv(raw_path)
        df = engineer_dataframe(df, config, target_type=target_type_override)
    else:
        processed_path = resolve_path(config["processed_output"])
        df = pd.read_csv(processed_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
    return apply_lookback(df, config)


def train_and_save(ticker, target_type_override=None):
    config = load_config(ticker)
    target_type = target_type_override or config["target_type"]
    df = load_training_frame(ticker, config, target_type_override=target_type_override)
    X, y, features = prepare_xy(df, config)
    split = int(len(df) * 0.8)

    model = RandomForestRegressor(**REGRESSOR_PARAMS)
    model.fit(X.iloc[:split], y.iloc[:split])

    bundle = {
        "ticker": ticker.upper(),
        "target_type": target_type,
        "features": features,
        "train_rows": split,
        "trained_at": datetime.now().isoformat(),
    }
    model_path = os.path.join(_artifacts_dir(), f"model_{ticker.upper()}.joblib")
    meta_path = os.path.join(_artifacts_dir(), f"model_{ticker.upper()}.json")
    joblib.dump({"model": model, "meta": bundle}, model_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    return model_path, bundle


def load_bundle(ticker):
    model_path = os.path.join(_artifacts_dir(), f"model_{ticker.upper()}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model for {ticker}. Run: python scripts/predict.py --train {ticker}"
        )
    return joblib.load(model_path)


def predict_latest(ticker):
    bundle = load_bundle(ticker)
    model = bundle["model"]
    meta = bundle["meta"]
    config = load_config(ticker)
    df = load_training_frame(ticker, config)
    X, _, _ = prepare_xy(df, config)
    latest = X.iloc[[-1]]
    pred = float(model.predict(latest)[0])
    return {
        "ticker": ticker.upper(),
        "target_type": meta["target_type"],
        "prediction": pred,
        "features": meta["features"],
    }
