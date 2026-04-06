import pandas as pd
import numpy as np
import sys
import json
import os

def run_feature_engineering(ticker):
    print(f"🚀 Starting Feature Engineering for: {ticker}")
    
    # 1. Path setup and Configuration Loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs.json')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        config = configs[ticker]
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error: Config for {ticker} not found! ({e})")
        return

    # 2. Define I/O Paths (Targeting the data/ directory)
    # Using os.path.join to ensure path compatibility on macOS M4
    input_path = os.path.join(current_dir, '..', config['data_source'])
    output_path = os.path.join(current_dir, '..', config['processed_output'])

    if not os.path.exists(input_path):
        print(f"❌ Error: Input artifact {input_path} not found!")
        print(f"💡 Tip: Execute 'python3 src/get_data.py {ticker}' first.")
        return
        
    # 3. Load Dataset and Sort by Time Series
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 4. Generate Base Indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 5. Dynamic Feature Loading based on configs.json
    active_features = config['features']
    
    if 'MA60' in active_features:
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
    if 'MA_Ratio' in active_features:
        df['MA_Ratio'] = df['MA5'] / df['MA20']
        
    if 'Vol_MA5' in active_features:
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
        
    if 'Vol_Norm' in active_features:
        # Volume Normalization: Scales volume down to prevent gradient issues
        df['Vol_Norm'] = df['Volume'].rolling(window=5).mean() / 1_000_000 

    if 'Volatility' in active_features or 'Volatility_20' in active_features:
        # Calculate volatility based on Target Type (Returns vs. Absolute Price)
        if config['target_type'] in ['Log_Return', 'Simple_Return']:
            temp_ret = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility_20'] = temp_ret.rolling(window=20).std()
        else:
            df['Volatility'] = df['Close'].rolling(window=20).std()

    if 'RSI' in active_features:
        # Relative Strength Index calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    if 'Bias' in active_features:
        # Price Bias: Measures deviation from the 5-day moving average
        df['Bias'] = (df['Close'] - df['MA5']) / df['MA5']

    # 6. Target Variable Generation (The core of the prediction logic)
    target_type = config['target_type']
    
    if target_type == 'Log_Return':
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Target'] = df['Log_Return'].shift(-1)
    elif target_type == 'Simple_Return':
        df['Daily_Return'] = df['Close'].pct_change()
        df['Target'] = df['Daily_Return'].shift(-1)
    else:
        # Predict absolute price (e.g., for high-value assets like AVGO)
        df['Target'] = df['Close'].shift(-1)

    # 7. Final Artifact Persistence (Cleanup NaNs and Save)
    df = df.dropna()
    df.to_csv(output_path, index=False)
    
    print("-" * 40)
    print(f"✅ {ticker} Feature Engineering SUCCESS!")
    print(f"📊 Target Generation Mode: {target_type}")
    print(f"📊 Processed Data Saved: {output_path}")
    print(f"📊 Final Dataset Size: {len(df)} samples")
    print("-" * 40)

if __name__ == "__main__":
    # Usage: python3 src/3_feature_engineering.py MU
    # Defaults to GOOG if no argument provided
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    run_feature_engineering(target_ticker)