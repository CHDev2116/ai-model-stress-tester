import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from pipeline.reports import prune_reports  # noqa: E402

STOCKS = ["AVGO", "GOOG", "MU", "NVDA"]

METRICS_PATTERN = re.compile(r"^METRICS_JSON:(.+)$", re.MULTILINE)


def parse_metrics(stdout):
    match = METRICS_PATTERN.search(stdout)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def run_benchmark(skip_ingestion=False):
    report_dir = os.path.join(CURRENT_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)

    results_summary = {}
    metrics_by_ticker = {}
    any_failure = False

    for ticker in STOCKS:
        print(f"\n🚀 Processing Pipeline for: {ticker}...")
        stock_success = True
        ticker_metrics = None

        steps = [
            ("Data_Ingestion", "1_get_real_data.py"),
            ("Feature_Engineering", "3_feature_engineering.py"),
            ("Model_Training", "4_train_model.py"),
        ]
        if skip_ingestion:
            steps = [s for s in steps if s[0] != "Data_Ingestion"]

        for step_label, script_name in steps:
            script_path = os.path.join(SRC_DIR, script_name)
            expected_csv = os.path.join(CURRENT_DIR, "data", f"{ticker.lower()}_assets.csv")
            had_file_before = False
            before_mtime = None
            if step_label == "Data_Ingestion":
                had_file_before = os.path.exists(expected_csv)
                before_mtime = os.path.getmtime(expected_csv) if had_file_before else None

            print(f"📦 Executing {step_label}...")
            step_result = subprocess.run(
                [sys.executable, script_path, ticker],
                capture_output=True,
                text=True,
                cwd=CURRENT_DIR,
            )
            combined = step_result.stdout + step_result.stderr
            if step_result.stdout:
                print(step_result.stdout, end="")
            if step_result.stderr:
                print(step_result.stderr, end="")

            if step_result.returncode != 0:
                print(f"❌ {ticker} pipeline failed during {step_label}: exit_code={step_result.returncode}")
                stock_success = False
                any_failure = True
                break

            if step_label == "Data_Ingestion":
                no_market_data = "No market data found" in combined
                has_file_after = os.path.exists(expected_csv)
                after_mtime = os.path.getmtime(expected_csv) if has_file_after else None
                file_not_updated = had_file_before and before_mtime == after_mtime
                if no_market_data or (not has_file_after) or file_not_updated:
                    print(f"❌ Critical Error: {ticker} ingestion did not produce fresh raw data.")
                    stock_success = False
                    any_failure = True
                    break

            if step_label == "Model_Training":
                ticker_metrics = parse_metrics(step_result.stdout)
                if ticker_metrics is None:
                    print(f"❌ {ticker} training finished without METRICS_JSON output.")
                    stock_success = False
                    any_failure = True
                    break

        results_summary[ticker] = "Success" if stock_success else "Failed"
        if ticker_metrics:
            metrics_by_ticker[ticker] = ticker_metrics

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"performance_report_{timestamp}.json")

    final_report = {
        "timestamp": timestamp,
        "results": results_summary,
        "metrics": metrics_by_ticker,
        "system_info": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "status": "Pipeline Run Complete" if not any_failure else "Pipeline Run Completed With Failures",
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(final_report, handle, indent=2)

    prune_reports(report_dir, keep_latest=10)

    print(f"\n📊 Benchmark Audit Report saved to: {report_path}")
    print(f"📈 Execution Summary: {results_summary}")
    if metrics_by_ticker:
        print(f"📉 Metrics: {json.dumps(metrics_by_ticker, indent=2)}")

    if any_failure:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock modeling benchmark pipeline.")
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip online data download and use existing local data files.",
    )
    args = parser.parse_args()
    run_benchmark(skip_ingestion=args.skip_ingestion)
