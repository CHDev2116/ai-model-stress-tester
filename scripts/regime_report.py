#!/usr/bin/env python3
"""Per-regime (high/low vol) evaluation report."""
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.config import load_all_configs, load_config  # noqa: E402
from pipeline.paths import resolve_path  # noqa: E402
from pipeline.regime import evaluate_by_regime  # noqa: E402
from pipeline.training import load_training_frame  # noqa: E402


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or list(load_all_configs().keys())
    report = {"timestamp": datetime.now().isoformat(), "tickers": {}}

    for ticker in tickers:
        config = load_config(ticker)
        df = load_training_frame(ticker, config)
        report["tickers"][ticker] = evaluate_by_regime(df, config)
        print(f"{ticker}: {json.dumps(report['tickers'][ticker]['regimes'])}")

    out = resolve_path(f"reports/regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"📄 Saved: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
