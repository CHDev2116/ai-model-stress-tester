import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from pipeline.config import fail, load_config
from pipeline.paths import resolve_path


def fetch_stock_data(ticker):
    print(f"🚀 Starting Data Ingestion for: {ticker}")

    try:
        config = load_config(ticker)
        save_path = resolve_path(config["data_source"])
    except KeyError:
        fail(
            f"❌ Error: Ticker {ticker} is not in src/configs.json.\n"
            f"💡 Add a config block or use one of: AVGO, GOOG, MU, NVDA."
        )
    except FileNotFoundError as exc:
        fail(f"❌ Error: configs.json not found ({exc})")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    end_date = (datetime.now() - timedelta(days=1)).date()
    try:
        start_date = end_date.replace(year=end_date.year - 2)
    except ValueError:
        start_date = end_date.replace(month=2, day=28, year=end_date.year - 2)

    data = yf.download(ticker, start=start_date.isoformat(), end=end_date.isoformat())

    if data.empty:
        fail(f"❌ Error: No market data found for {ticker}")

    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    required_cols = ["Date", "Close", "Volume"]
    optional_cols = ["High", "Low"]
    selected_cols = required_cols + [c for c in optional_cols if c in df.columns]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        fail(f"❌ Error: Missing required columns from data source: {missing_required}")

    df = df[selected_cols]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna()

    df.to_csv(save_path, index=False)
    print(f"✅ Success! Raw data saved to: {save_path}")
    print(f"📊 Dataset Info: Latest Date = {df['Date'].max().date()} | Total Rows = {len(df)}\n")


if __name__ == "__main__":
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    fetch_stock_data(target_ticker)
    sys.exit(0)
