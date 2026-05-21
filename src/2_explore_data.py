import os
import sys

import pandas as pd

from pipeline.config import fail, load_config
from pipeline.paths import resolve_path


def explore_data(ticker):
    try:
        config = load_config(ticker)
        file_path = resolve_path(config["data_source"])
    except KeyError:
        fail(
            f"❌ Error: Ticker {ticker} is not in src/configs.json.\n"
            f"💡 Run: python src/1_get_real_data.py <TICKER> after adding config."
        )
    except FileNotFoundError as exc:
        fail(f"❌ Error: configs.json not found ({exc})")

    if not os.path.exists(file_path):
        fail(
            f"❌ Error: Data file not found at {file_path}\n"
            f"💡 Tip: Execute 'python src/1_get_real_data.py {ticker}' first."
        )

    df = pd.read_csv(file_path)
    print(f"\n✅ Successfully loaded {ticker} dataset!")
    print("-" * 40)
    print("Top 5 Rows (Preview):")
    print(df.head())
    print("\n📊 Statistical Summary (Profiling):")
    print(df.describe())
    print("-" * 40)


if __name__ == "__main__":
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    explore_data(target_ticker)
    sys.exit(0)
