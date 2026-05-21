"""Compare Price vs return targets on the same walk-forward setup."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.config import fail, load_all_configs  # noqa: E402
from pipeline.features import engineer_dataframe  # noqa: E402
from pipeline.paths import resolve_path  # noqa: E402
from pipeline.training import evaluate_dataframe  # noqa: E402


def load_raw(ticker, config):
    path = resolve_path(config["data_source"])
    if not os.path.exists(path):
        fail(f"❌ Error: Raw data not found at {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


def evaluate(ticker, target_type):
    config = load_all_configs()[ticker.upper()]
    df = engineer_dataframe(load_raw(ticker, config), config, target_type=target_type)
    metrics = evaluate_dataframe(df, config, target_type=target_type)
    return metrics


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or ["AVGO", "GOOG", "MU", "NVDA"]
    configs = load_all_configs()

    print("ticker,baseline_target,baseline_mae,optimized_target,optimized_mae,optimized_direction_acc")
    for ticker in tickers:
        optimized_type = configs[ticker]["target_type"]
        baseline = evaluate(ticker, "Price")
        optimized = evaluate(ticker, optimized_type)
        dir_str = ""
        if optimized["direction_accuracy"] is not None:
            dir_str = f"{optimized['direction_accuracy'] * 100:.2f}%"
        print(
            f"{ticker},Price,{baseline['mae']:.6f},{optimized_type},"
            f"{optimized['mae']:.6f},{dir_str}"
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
