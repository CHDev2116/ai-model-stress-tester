import os
import pandas as pd
import sys
import json

def explore_data(ticker):
    # 1. Define local paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs.json')

    # 2. Retrieve data source path from configs.json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        # Access the specific data source path (e.g., data/mu_assets.csv)
        relative_path = configs[ticker]['data_source']
        file_path = os.path.join(current_dir, '..', relative_path)
        
    except (FileNotFoundError, KeyError):
        print(f"⚠️ Warning: Configuration for {ticker} not found. Attempting default fallback path...")
        file_path = os.path.join(current_dir, '..', 'data', f'{ticker.lower()}_assets.csv')

    # 3. Validation: Check if the raw data file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Data file not found at {file_path}")
        print(f"💡 Tip: Please execute 'python3 src/get_data.py {ticker}' first to download data.")
        return

    # 4. Load and Profile Data
    df = pd.read_csv(file_path)
    print(f"\n✅ Successfully loaded {ticker} dataset!")
    print("-" * 40)
    print("Top 5 Rows (Preview):")
    print(df.head())

    print("\n📊 Statistical Summary (Profiling):")
    print(df.describe())
    print("-" * 40)

if __name__ == "__main__":
    # Usage: python3 src/2_explore_data.py MU
    # Defaults to GOOG if no ticker is provided via CLI
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    explore_data(target_ticker)