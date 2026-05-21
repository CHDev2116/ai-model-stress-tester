#!/usr/bin/env python3
"""Rolling walk-forward fold report (JSON) per ticker."""
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.config import load_all_configs  # noqa: E402
from pipeline.paths import resolve_path  # noqa: E402
from pipeline.training import load_training_frame  # noqa: E402
from pipeline.validation import rolling_walk_forward_metrics  # noqa: E402


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or list(load_all_configs().keys())
    report = {"timestamp": datetime.now().isoformat(), "tickers": {}}

    for ticker in tickers:
        from pipeline.config import load_config

        config = load_config(ticker)
        df = load_training_frame(ticker, config)
        report["tickers"][ticker] = rolling_walk_forward_metrics(df, config, n_folds=5)
        s = report["tickers"][ticker]
        print(
            f"{ticker}: folds={s['n_folds']} mae_mean={s['mae_mean']:.6f} "
            f"mae_std={s['mae_std']:.6f}"
        )

    out = resolve_path(
        f"reports/walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"📄 Saved: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
