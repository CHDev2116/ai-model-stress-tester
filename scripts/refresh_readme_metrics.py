#!/usr/bin/env python3
"""Refresh README Results table from compare_targets.py (in-place markers)."""
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
sys.path.insert(0, str(ROOT / "src"))

from pipeline.config import load_all_configs  # noqa: E402

# Import compare logic
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "compare_targets", ROOT / "scripts" / "compare_targets.py"
)
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)


def latest_data_date():
    import pandas as pd

    dates = []
    for ticker, cfg in load_all_configs().items():
        path = ROOT / cfg["data_source"]
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["Date"])
        dates.append(pd.to_datetime(df["Date"]).max())
    if not dates:
        return datetime.now().date().isoformat()
    return max(dates).date().isoformat()


def build_table_rows():
    rows = []
    configs = load_all_configs()
    for ticker in configs:
        baseline = _compare.evaluate(ticker, "Price")
        opt_type = configs[ticker]["target_type"]
        optimized = _compare.evaluate(ticker, opt_type)
        dir_acc = optimized.get("direction_accuracy")
        dir_str = f"{dir_acc * 100:.1f}%" if dir_acc is not None else "—"
        opt_mae = (
            f"**{optimized['mae'] * 100:.2f}%**"
            if "Return" in opt_type
            else f"**{optimized['mae']:.2f}**"
        )
        rows.append(
            f"| {ticker} | `Price` | {baseline['mae']:.2f} | `{opt_type}` | {opt_mae} | {dir_str} |"
        )
    return rows


def refresh_readme():
    text = README.read_text(encoding="utf-8")
    data_date = latest_data_date()
    refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    intro = (
        f"Measured on four high-volatility tickers (`AVGO`, `GOOG`, `MU`, `NVDA`) "
        f"with Yahoo Finance data through **{data_date}** (~480 samples each after feature warmup).  \n"
        f"*Metrics last refreshed: {refreshed_at} via `python scripts/refresh_readme_metrics.py`.*"
    )

    table_header = (
        "| Ticker | Baseline target | Baseline MAE (price units) | Optimized target | Optimized MAE | Direction accuracy |\n"
        "|--------|-----------------|----------------------------|------------------|---------------|----------------------|"
    )
    table_body = "\n".join(build_table_rows())
    block = f"{intro}\n\n{table_header}\n{table_body}"

    pattern = re.compile(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
        re.DOTALL,
    )
    replacement = f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->"
    if not pattern.search(text):
        fail_msg = "README missing <!-- RESULTS:START --> / <!-- RESULTS:END --> markers"
        print(f"❌ {fail_msg}")
        sys.exit(1)

    text = pattern.sub(replacement, text)
    README.write_text(text, encoding="utf-8")
    print(f"✅ Updated {README} (data through {data_date})")


if __name__ == "__main__":
    refresh_readme()
    sys.exit(0)
