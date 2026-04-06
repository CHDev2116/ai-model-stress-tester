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
from sklearn.metrics import mean_absolute_error

def train_model_with_profiling(ticker):
    # 1. Initialize Hardware Profiling
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024  # Unit: MB
    start_time = time.time()

    print(f"🤖 Profiling Model Training for: {ticker} on Apple M4...")

    # 2. Load Configuration and Paths
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
        print(f"❌ Error: Processed data artifact not found at {input_path}")
        return
        
    df = pd.read_csv(input_path)
    
    # 3. Data Preparation (Predicting Market Returns/Prices)
    features = [f for f in config['features'] if f in df.columns]
    X = df[features]
    y = df['Target']
    
    # 4. Time-Series Split (Last 20% reserved for testing)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Model Training (Leveraging M4 multi-core performance)
    # n_jobs=-1 uses all available CPU cores
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 6. Inference and Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # 7. Visualization (Includes MAE and Hardware Metadata)
    plt.figure(figsize=(12, 7))
    plt.plot(y_test.values, label='Actual (Market Ground Truth)', color='#1f77b4', alpha=0.7)
    plt.plot(preds, label='AI Predicted Signal', color='#d62728', linestyle='--')
    
    # Information Box (MAE + Metadata)
    error_display = f"MAE: {mae:.6f}"
    if 'Return' in config['target_type']:
        error_display += f" ({mae*100:.2f}%)"
    
    stats_text = f"Target: {config['target_type']}\n{error_display}\nHardware: Apple M4 Chip"
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.title(f'{ticker} Performance Analysis - Predictive Modeling', fontsize=14)
    plt.xlabel('Trading Days (Test Period)', fontsize=12)
    plt.ylabel('Value (Return or Absolute Price)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)

    # 8. Automated Persistence with Timestamped Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(current_dir, '..', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    chart_filename = f'chart_{ticker}_{timestamp}.png'
    chart_path = os.path.join(report_dir, chart_filename)
    
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # 9. Performance Profiling Summary
    duration = time.time() - start_time
    mem_used = (process.memory_info().rss / 1024 / 1024) - start_mem

    print("-" * 45)
    print(f"✅ {ticker} TRAINING & PROFILING SUCCESSFUL!")
    print(f"⏱️ Latency: {duration:.2f}s | 🧠 Memory Delta: {mem_used:.2f}MB")
    print(f"🏆 Metric: {error_display}")
    print(f"🖼️ Artifact Saved: {chart_path}")
    print("-" * 45)

if __name__ == "__main__":
    # Usage: python3 src/4_train_model.py MU
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    train_model_with_profiling(target_ticker)