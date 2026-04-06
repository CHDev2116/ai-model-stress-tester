import yfinance as yf
import pandas as pd
import os
import sys
import json

def fetch_stock_data(ticker):
    print(f"🚀 Starting Data Ingestion for: {ticker}")

    # 1. Load configuration (Ensure configs.json is accessible)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs.json')
    
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        config = configs[ticker]
        # Map path relative to the project root
        save_path = os.path.join(current_dir, '..', config['data_source'])
    except (FileNotFoundError, KeyError):
        print(f"⚠️ Warning: Config for {ticker} not found. Using default fallback path.")
        save_path = os.path.join(current_dir, '..', 'data', f"{ticker.lower()}_assets.csv")

    # 2. Ensure the data directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 3. Download market data (From 2023 to present to ensure sufficient MA60 history)
    # Note: Current date set to 2026-04-06
    data = yf.download(ticker, start="2023-01-01", end="2026-04-06")
    
    if data.empty:
        print(f"❌ Error: No market data found for {ticker}")
        return

    # 4. Handle yfinance MultiIndex column issues
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 5. Data Cleaning and Formatting
    df = df.reset_index()
    # Ensure standard OHLCV columns exist
    df = df[['Date', 'Close', 'Volume']]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna()

    # 6. Persistence
    df.to_csv(save_path, index=False)
    print(f"✅ Success! Raw data saved to: {save_path}")
    print(f"📊 Dataset Info: Latest Date = {df['Date'].max().date()} | Total Rows = {len(df)}\n")

if __name__ == "__main__":
    # Usage: python3 src/get_data.py MU
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    fetch_stock_data(target_ticker)