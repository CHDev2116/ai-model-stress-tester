#!/usr/bin/env python3
"""CLI: train artifact bundle or predict latest row."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.inference import predict_latest, train_and_save  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train or run local inference.")
    parser.add_argument("ticker", nargs="?", default="GOOG", help="Ticker symbol")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fit and save model under artifacts/",
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    if args.train:
        model_path, meta = train_and_save(ticker)
        print(f"✅ Saved model: {model_path}")
        print(json.dumps(meta, indent=2))
        return

    try:
        result = predict_latest(ticker)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
    sys.exit(0)
