#!/usr/bin/env python3
"""Generate committed NVDA Price-target failure chart for README."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.paths import resolve_path  # noqa: E402
from pipeline.training import run_training  # noqa: E402

STABLE_CHART = "reports/chart_NVDA_failure.png"


def main():
    ticker = "NVDA"
    stable_path = resolve_path(STABLE_CHART)
    os.makedirs(os.path.dirname(stable_path), exist_ok=True)

    run_training(
        ticker,
        write_charts=True,
        target_type_override="Price",
        signal_chart_path=stable_path,
        skip_prune=True,
    )
    print(f"✅ Failure chart saved: {stable_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
