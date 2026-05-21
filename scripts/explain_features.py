#!/usr/bin/env python3
"""SHAP feature importance report per ticker."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.config import load_config  # noqa: E402
from pipeline.explain import save_shap_report  # noqa: E402
from pipeline.training import load_training_frame  # noqa: E402


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    config = load_config(ticker)
    df = load_training_frame(ticker, config)
    try:
        path, summary = save_shap_report(ticker, df, config)
    except ImportError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    print(f"✅ SHAP report: {path}")
    print(f"Top features: {summary['top_features']}")


if __name__ == "__main__":
    main()
