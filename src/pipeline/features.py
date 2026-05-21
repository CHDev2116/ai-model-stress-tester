import os
import sys

import numpy as np
import pandas as pd

from pipeline.config import fail, load_config
from pipeline.paths import resolve_path


def apply_lookback(df, config):
    if not config.get("lookback_days") or "Date" not in df.columns:
        return df
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    cutoff = out["Date"].max() - pd.Timedelta(days=int(config["lookback_days"]))
    return out[out["Date"] >= cutoff].copy()


def engineer_dataframe(df, config, target_type=None):
    """Build indicators and Target column. Optional target_type overrides config."""
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date")

    target_type = target_type or config["target_type"]
    active_features = config["features"]

    out["MA5"] = out["Close"].rolling(window=5).mean()
    out["MA20"] = out["Close"].rolling(window=20).mean()

    if "Daily_Return" in active_features:
        out["Daily_Return"] = out["Close"].pct_change()

    if "Vol_Ratio" in active_features:
        out["Vol_Ratio"] = out["Volume"] / out["Volume"].rolling(window=5).mean()

    if "Price_Range" in active_features:
        if {"High", "Low"}.issubset(out.columns):
            out["Price_Range"] = (out["High"] - out["Low"]) / out["Close"]
        else:
            print(
                "⚠️ Warning: High/Low columns missing, "
                "using abs(close pct_change) as Price_Range proxy."
            )
            out["Price_Range"] = out["Close"].pct_change().abs()

    if "MA60" in active_features:
        out["MA60"] = out["Close"].rolling(window=60).mean()

    if "MA_Ratio" in active_features:
        out["MA_Ratio"] = out["MA5"] / out["MA20"]

    if "Vol_MA5" in active_features:
        out["Vol_MA5"] = out["Volume"].rolling(window=5).mean()

    if "Vol_Norm" in active_features:
        out["Vol_Norm"] = out["Volume"].rolling(window=5).mean() / 1_000_000

    if "Volatility_20" in active_features:
        temp_ret = np.log(out["Close"] / out["Close"].shift(1))
        out["Volatility_20"] = temp_ret.rolling(window=20).std()
    elif "Volatility" in active_features:
        out["Volatility"] = out["Close"].rolling(window=20).std()

    if "RSI" in active_features:
        delta = out["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        out["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    if "Bias" in active_features:
        out["Bias"] = (out["Close"] - out["MA5"]) / out["MA5"]

    if target_type == "Log_Return":
        out["Log_Return"] = np.log(out["Close"] / out["Close"].shift(1))
        out["Target"] = out["Log_Return"].shift(-1)
    elif target_type == "Simple_Return":
        out["Daily_Return_Target"] = out["Close"].pct_change()
        out["Target"] = out["Daily_Return_Target"].shift(-1)
    else:
        out["Target"] = out["Close"].shift(-1)

    return out.dropna()


def run_feature_engineering(ticker):
    print(f"🚀 Starting Advanced Feature Engineering for: {ticker}")
    try:
        config = load_config(ticker)
    except (FileNotFoundError, KeyError) as exc:
        fail(f"❌ Error: Config for {ticker} not found! ({exc})")

    input_path = resolve_path(config["data_source"])
    output_path = resolve_path(config["processed_output"])

    if not os.path.exists(input_path):
        fail(
            f"❌ Error: Input artifact {input_path} not found!\n"
            f"💡 Tip: Execute 'python src/1_get_real_data.py {ticker}' first."
        )

    df = pd.read_csv(input_path)
    df = engineer_dataframe(df, config)
    df = apply_lookback(df, config)

    if len(df) < 50:
        fail(f"❌ Error: Not enough samples after feature engineering for {ticker}.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("-" * 45)
    print(f"✅ {ticker} FEATURE ENGINEERING SUCCESS!")
    print(f"📊 Target Mode: {config['target_type']}")
    print(f"📊 Active Features: {len(config['features'])}")
    print(f"📊 Final Dataset Size: {len(df)} samples")
    print("-" * 45)
