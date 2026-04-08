import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
import sys
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

def prune_reports(report_dir, keep_latest=10):
    files = []
    for name in os.listdir(report_dir):
        path = os.path.join(report_dir, name)
        if os.path.isfile(path):
            files.append(path)
    files.sort(key=os.path.getmtime, reverse=True)
    for old_path in files[keep_latest:]:
        os.remove(old_path)

def build_price_path(start_price, returns, target_type):
    if target_type == 'Log_Return':
        growth = np.exp(np.cumsum(returns))
    else:
        growth = np.cumprod(1 + returns)
    return start_price * growth

def walk_forward_predictions(X, y, split, model_params, retrain_every=20):
    X_test = X.iloc[split:]
    preds = np.zeros(len(X_test))
    for start in range(0, len(X_test), retrain_every):
        train_end = split + start
        chunk_end = min(start + retrain_every, len(X_test))
        model = RandomForestRegressor(**model_params)
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        preds[start:chunk_end] = model.predict(X_test.iloc[start:chunk_end])
    return preds

def walk_forward_direction_probs(X, y, split, model_params, retrain_every=20):
    X_test = X.iloc[split:]
    probs = np.zeros(len(X_test))
    for start in range(0, len(X_test), retrain_every):
        train_end = split + start
        chunk_end = min(start + retrain_every, len(X_test))
        direction_target = (y.iloc[:train_end] > 0).astype(int)
        clf = RandomForestClassifier(**model_params)
        clf.fit(X.iloc[:train_end], direction_target)
        probs[start:chunk_end] = clf.predict_proba(X_test.iloc[start:chunk_end])[:, 1]
    return probs

def recent_drift_baseline(y, split, window=60):
    y_test = y.iloc[split:]
    baseline = np.zeros(len(y_test))
    for i in range(len(y_test)):
        train_end = split + i
        start = max(0, train_end - window)
        baseline[i] = float(y.iloc[start:train_end].mean())
    return baseline

def train_model_with_profiling(ticker):
    # 1. Initialize Hardware Profiling
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    print(f"🤖 Training Optimized Model for: {ticker} on Apple M4...")

    # 2. Load Configuration and Artifacts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs.json')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)[ticker]
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error: Configuration for {ticker} not found! ({e})")
        return
    
    input_path = os.path.join(current_dir, '..', config['processed_output'])
    if not os.path.exists(input_path):
        print(f"❌ Error: Processed data not found at {input_path}")
        return
        
    df = pd.read_csv(input_path)
    if 'Date' in df.columns and config.get('lookback_days'):
        df['Date'] = pd.to_datetime(df['Date'])
        cutoff = df['Date'].max() - pd.Timedelta(days=int(config['lookback_days']))
        df = df[df['Date'] >= cutoff].copy()
        if len(df) < 200:
            print(f"❌ Error: Not enough samples after lookback filter for {ticker}.")
            return
    
    # 3. Feature Selection (Sync with 3_feature_engineering.py)
    configured_features = config['features']
    missing_features = [f for f in configured_features if f not in df.columns]
    if missing_features:
        print(f"⚠️ Warning: Missing configured features for {ticker}: {missing_features}")
    features = [f for f in configured_features if f in df.columns]
    if not features:
        print(f"❌ Error: No valid features available for {ticker}.")
        return
    X = df[features]
    y = df['Target']
    
    # 4. Sequential Split (80% Train, 20% Test)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Hyperparameter Tuning for Volatility Capture
    # - n_estimators: Reduced slightly to prevent over-smoothing
    # - max_depth: Increased to allow trees to capture more complex patterns
    # - min_samples_leaf: Kept at 1 to maintain sensitivity to outliers
    # - max_features: 'sqrt' helps decorrelate trees for better variance capture
    model_params = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    
    # 6. Inference and Evaluation
    preds = walk_forward_predictions(X, y, split, model_params, retrain_every=20)
    if 'Return' in config['target_type']:
        direction_model_params = dict(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        up_probs = walk_forward_direction_probs(X, y, split, direction_model_params, retrain_every=20)
        train_up_prior = float((y_train > 0).mean())
        blended_up_probs = 0.7 * up_probs + 0.3 * train_up_prior
        signed_factor = (2 * blended_up_probs) - 1  # [-1, 1]
        preds = np.abs(preds) * signed_factor
        # Anchor predictions to recent market drift so strong uptrends are not flattened.
        drift_base = recent_drift_baseline(y, split, window=60)
        preds = 0.65 * preds + 0.35 * drift_base
    mae = mean_absolute_error(y_test, preds)
    directional_accuracy = None
    if 'Return' in config['target_type']:
        directional_accuracy = np.mean(np.sign(y_test.values) == np.sign(preds))

    # 7. Visualization with Ground Truth vs AI Signal
    plt.figure(figsize=(12, 7))
    plt.plot(y_test.values, label='Actual Market Data', color='#1f77b4', alpha=0.6, linewidth=1.5)
    plt.plot(preds, label='AI Predictive Signal', color='#d62728', linestyle='--', linewidth=1.2)
    
    # Metrics Overlay
    error_display = f"MAE: {mae:.6f}"
    if 'Return' in config['target_type']:
        error_display += f" ({mae*100:.2f}%)"
    direction_display = ""
    if directional_accuracy is not None:
        direction_display = f"\nDirection Acc: {directional_accuracy*100:.2f}%"
    
    stats_text = (
        f"Target: {config['target_type']}\n"
        f"{error_display}{direction_display}\n"
        "Config: Walk-Forward(20), RF-Reg + RF-Dir blend"
    )
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title(f'{ticker} Strategy Analysis - High Sensitivity Mode', fontsize=14)
    plt.xlabel('Test Samples (Timeline)', fontsize=12)
    plt.ylabel('Target Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.4)

    # 8. Report Generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(current_dir, '..', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    chart_path = os.path.join(report_dir, f'chart_{ticker}_{timestamp}.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # Return-based targets are hard to interpret visually; also show reconstructed price paths.
    price_path_chart = None
    if config['target_type'] in ['Log_Return', 'Simple_Return']:
        start_price = df['Close'].iloc[split]
        actual_path = build_price_path(start_price, y_test.values, config['target_type'])
        pred_path = build_price_path(start_price, preds, config['target_type'])

        plt.figure(figsize=(12, 7))
        plt.plot(actual_path, label='Actual Price Path (Reconstructed)', color='#1f77b4', alpha=0.7, linewidth=1.6)
        plt.plot(pred_path, label='Predicted Price Path (Reconstructed)', color='#d62728', linestyle='--', linewidth=1.4)
        plt.title(f'{ticker} Reconstructed Price Path from Return Predictions', fontsize=14)
        plt.xlabel('Test Samples (Timeline)', fontsize=12)
        plt.ylabel('Price Level (Reconstructed)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.gca().text(
            0.02, 0.95,
            f"Start Price: {start_price:.2f}\n{error_display}{direction_display}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        price_path_chart = os.path.join(report_dir, f'chart_{ticker}_{timestamp}_pricepath.png')
        plt.savefig(price_path_chart, dpi=150)
        plt.close()

    prune_reports(report_dir, keep_latest=10)

    # 9. Profiling Output
    duration = time.time() - start_time
    mem_used = (process.memory_info().rss / 1024 / 1024) - start_mem

    print("-" * 45)
    print(f"✅ {ticker} OPTIMIZED TRAINING SUCCESSFUL!")
    print(f"⏱️ Latency: {duration:.2f}s | 🧠 Memory Delta: {mem_used:.2f}MB")
    print(f"🏆 {error_display}")
    if directional_accuracy is not None:
        print(f"🎯 Directional Accuracy: {directional_accuracy*100:.2f}%")
    print(f"🖼️ Signal Chart: {chart_path}")
    if price_path_chart:
        print(f"🖼️ Price Path Chart: {price_path_chart}")
    print("-" * 45)

if __name__ == "__main__":
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    train_model_with_profiling(target_ticker)