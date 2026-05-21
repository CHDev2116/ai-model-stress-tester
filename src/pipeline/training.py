import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from pipeline.config import fail, load_config
from pipeline.features import apply_lookback
from pipeline.paths import resolve_path
from pipeline.reports import prune_reports

REGRESSOR_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

DIRECTION_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

MIN_SAMPLES = 200
TRAIN_FRACTION = 0.8
RETRAIN_EVERY = 20


def build_price_path(start_price, returns, target_type):
    if target_type == "Log_Return":
        growth = np.exp(np.cumsum(returns))
    else:
        growth = np.cumprod(1 + returns)
    return start_price * growth


def walk_forward_predictions(X, y, split, model_factory=None, retrain_every=RETRAIN_EVERY):
    if model_factory is None:
        model_factory = lambda: RandomForestRegressor(**REGRESSOR_PARAMS)
    X_test = X.iloc[split:]
    preds = np.zeros(len(X_test))
    for start in range(0, len(X_test), retrain_every):
        train_end = split + start
        chunk_end = min(start + retrain_every, len(X_test))
        model = model_factory()
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        preds[start:chunk_end] = model.predict(X_test.iloc[start:chunk_end])
    return preds


def _positive_class_proba(clf, X_chunk):
    proba = clf.predict_proba(X_chunk)
    if proba.shape[1] == 1:
        return proba[:, 0] if clf.classes_[0] == 1 else np.zeros(len(X_chunk))
    pos_idx = 1 if 1 in clf.classes_ else 0
    return proba[:, pos_idx]


def walk_forward_direction_probs(X, y, split, model_params=None, retrain_every=RETRAIN_EVERY):
    model_params = model_params or DIRECTION_PARAMS
    X_test = X.iloc[split:]
    probs = np.zeros(len(X_test))
    for start in range(0, len(X_test), retrain_every):
        train_end = split + start
        chunk_end = min(start + retrain_every, len(X_test))
        direction_target = (y.iloc[:train_end] > 0).astype(int)
        clf = RandomForestClassifier(**model_params)
        clf.fit(X.iloc[:train_end], direction_target)
        probs[start:chunk_end] = _positive_class_proba(
            clf, X_test.iloc[start:chunk_end]
        )
    return probs


def recent_drift_baseline(y, split, window=60):
    y_test = y.iloc[split:]
    baseline = np.zeros(len(y_test))
    for i in range(len(y_test)):
        train_end = split + i
        start = max(0, train_end - window)
        baseline[i] = float(y.iloc[start:train_end].mean())
    return baseline


def prepare_xy(df, config):
    configured_features = config["features"]
    missing_features = [f for f in configured_features if f not in df.columns]
    if missing_features:
        print(f"⚠️ Warning: Missing configured features: {missing_features}")
    features = [f for f in configured_features if f in df.columns]
    if not features:
        fail("❌ Error: No valid features available.")
    return df[features], df["Target"], features


def predict_targets(X, y, split, target_type, model_factory=None):
    preds = walk_forward_predictions(X, y, split, model_factory=model_factory)
    if "Return" not in target_type:
        return preds

    up_probs = walk_forward_direction_probs(X, y, split)
    train_up_prior = float((y.iloc[:split] > 0).mean())
    blended_up_probs = 0.7 * up_probs + 0.3 * train_up_prior
    signed_factor = (2 * blended_up_probs) - 1
    preds = np.abs(preds) * signed_factor
    drift_base = recent_drift_baseline(y, split, window=60)
    return 0.65 * preds + 0.35 * drift_base


def evaluate_dataframe(df, config, target_type=None):
    """Walk-forward evaluation; returns metrics dict (no charts)."""
    target_type = target_type or config["target_type"]
    df = apply_lookback(df, config)
    if len(df) < MIN_SAMPLES:
        fail(f"❌ Error: Not enough samples ({len(df)}) after lookback filter.")

    X, y, feature_names = prepare_xy(df, config)
    split = int(len(df) * TRAIN_FRACTION)
    y_test = y.iloc[split:]
    preds = predict_targets(X, y, split, target_type)

    mae = float(mean_absolute_error(y_test, preds))
    direction_acc = None
    if "Return" in target_type:
        direction_acc = float(np.mean(np.sign(y_test.values) == np.sign(preds)))

    return {
        "target_type": target_type,
        "mae": mae,
        "mae_pct": mae * 100 if "Return" in target_type else None,
        "direction_accuracy": direction_acc,
        "n_samples": len(df),
        "n_test": len(y_test),
        "features": feature_names,
    }


def _setup_matplotlib():
    import matplotlib

    cache_dir = os.path.join(resolve_path("."), ".matplotlib-cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)


def load_training_frame(ticker, config, target_type_override=None):
    from pipeline.features import engineer_dataframe

    if target_type_override:
        raw_path = resolve_path(config["data_source"])
        if not os.path.exists(raw_path):
            fail(f"❌ Error: Raw data not found at {raw_path}")
        df = pd.read_csv(raw_path)
        df = engineer_dataframe(df, config, target_type=target_type_override)
    else:
        input_path = resolve_path(config["processed_output"])
        if not os.path.exists(input_path):
            fail(f"❌ Error: Processed data not found at {input_path}")
        df = pd.read_csv(input_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
    return apply_lookback(df, config)


def _save_charts(
    ticker,
    df,
    config,
    y_test,
    preds,
    mae,
    direction_accuracy,
    report_dir,
    timestamp,
    target_type=None,
    signal_chart_path=None,
):
    import matplotlib.pyplot as plt

    target_type = target_type or config["target_type"]
    error_display = f"MAE: {mae:.6f}"
    if "Return" in target_type:
        error_display += f" ({mae * 100:.2f}%)"
    direction_display = ""
    if direction_accuracy is not None:
        direction_display = f"\nDirection Acc: {direction_accuracy * 100:.2f}%"

    plt.figure(figsize=(12, 7))
    plt.plot(y_test.values, label="Actual Market Data", color="#1f77b4", alpha=0.6, linewidth=1.5)
    plt.plot(preds, label="AI Predictive Signal", color="#d62728", linestyle="--", linewidth=1.2)
    stats_text = (
        f"Target: {target_type}\n"
        f"{error_display}{direction_display}\n"
        "Config: Walk-Forward(20), RF-Reg + RF-Dir blend"
    )
    plt.gca().text(
        0.02,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    title_suffix = " (Price Target Failure)" if target_type == "Price" else ""
    plt.title(f"{ticker} Strategy Analysis{title_suffix}", fontsize=14)
    plt.xlabel("Test Samples (Timeline)", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.4)

    chart_path = signal_chart_path or os.path.join(
        report_dir, f"chart_{ticker}_{timestamp}.png"
    )
    plt.savefig(chart_path, dpi=150)
    plt.close()

    price_path_chart = None
    split = int(len(df) * TRAIN_FRACTION)
    if target_type in ["Log_Return", "Simple_Return"]:
        start_price = df["Close"].iloc[split]
        actual_path = build_price_path(start_price, y_test.values, target_type)
        pred_path = build_price_path(start_price, preds, target_type)

        plt.figure(figsize=(12, 7))
        plt.plot(
            actual_path,
            label="Actual Price Path (Reconstructed)",
            color="#1f77b4",
            alpha=0.7,
            linewidth=1.6,
        )
        plt.plot(
            pred_path,
            label="Predicted Price Path (Reconstructed)",
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
        )
        plt.title(
            f"{ticker} Reconstructed Price Path from Return Predictions",
            fontsize=14,
        )
        plt.xlabel("Test Samples (Timeline)", fontsize=12)
        plt.ylabel("Price Level (Reconstructed)", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.gca().text(
            0.02,
            0.95,
            f"Start Price: {start_price:.2f}\n{error_display}{direction_display}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        price_path_chart = os.path.join(
            report_dir, f"chart_{ticker}_{timestamp}_pricepath.png"
        )
        plt.savefig(price_path_chart, dpi=150)
        plt.close()

    return chart_path, price_path_chart


def run_training(
    ticker,
    write_charts=True,
    target_type_override=None,
    signal_chart_path=None,
    skip_prune=False,
):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    print(f"🤖 Training model for: {ticker} ({platform.machine()})...")

    try:
        config = load_config(ticker)
    except (FileNotFoundError, KeyError) as exc:
        fail(f"❌ Error: Configuration for {ticker} not found! ({exc})")

    df = load_training_frame(ticker, config, target_type_override=target_type_override)
    if len(df) < MIN_SAMPLES:
        fail(f"❌ Error: Not enough samples after lookback filter for {ticker}.")

    X, y, _ = prepare_xy(df, config)
    split = int(len(df) * TRAIN_FRACTION)
    y_test = y.iloc[split:]
    target_type = target_type_override or config["target_type"]
    preds = predict_targets(X, y, split, target_type)

    mae = float(mean_absolute_error(y_test, preds))
    direction_accuracy = None
    if "Return" in target_type:
        direction_accuracy = float(np.mean(np.sign(y_test.values) == np.sign(preds)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = resolve_path("reports")
    os.makedirs(report_dir, exist_ok=True)

    chart_path = None
    price_path_chart = None
    if write_charts:
        _setup_matplotlib()
        chart_path, price_path_chart = _save_charts(
            ticker,
            df,
            config,
            y_test,
            preds,
            mae,
            direction_accuracy,
            report_dir,
            timestamp,
            target_type=target_type,
            signal_chart_path=signal_chart_path,
        )
        if not skip_prune:
            prune_reports(report_dir, keep_latest=10)

    duration = time.time() - start_time
    mem_used = (process.memory_info().rss / 1024 / 1024) - start_mem

    metrics = {
        "ticker": ticker.upper(),
        "target_type": target_type,
        "mae": mae,
        "mae_pct": round(mae * 100, 4) if "Return" in target_type else None,
        "direction_accuracy": direction_accuracy,
        "latency_sec": round(duration, 2),
        "memory_delta_mb": round(mem_used, 2),
        "n_samples": len(df),
        "n_test": len(y_test),
        "chart_path": chart_path,
        "price_path_chart": price_path_chart,
        "timestamp": timestamp,
    }

    metrics_path = os.path.join(report_dir, f"metrics_{ticker.upper()}_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    error_display = f"MAE: {mae:.6f}"
    if "Return" in target_type:
        error_display += f" ({mae * 100:.2f}%)"

    print("-" * 45)
    print(f"✅ {ticker} TRAINING SUCCESSFUL!")
    print(f"⏱️ Latency: {duration:.2f}s | 🧠 Memory Delta: {mem_used:.2f}MB")
    print(f"🏆 {error_display}")
    if direction_accuracy is not None:
        print(f"🎯 Directional Accuracy: {direction_accuracy * 100:.2f}%")
    if chart_path:
        print(f"🖼️ Signal Chart: {chart_path}")
    if price_path_chart:
        print(f"🖼️ Price Path Chart: {price_path_chart}")
    print(f"📄 Metrics JSON: {metrics_path}")
    print("METRICS_JSON:" + json.dumps(metrics))
    print("-" * 45)

    return metrics
