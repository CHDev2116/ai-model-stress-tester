import subprocess
import os
import json
import sys
from datetime import datetime

# Configuration
STOCKS = ['AVGO', 'GOOG', 'MU', 'NVDA']
current_dir = os.path.dirname(os.path.abspath(__file__))

def prune_reports(report_dir, keep_latest=10):
    files = []
    for name in os.listdir(report_dir):
        path = os.path.join(report_dir, name)
        if os.path.isfile(path):
            files.append(path)
    files.sort(key=os.path.getmtime, reverse=True)
    for old_path in files[keep_latest:]:
        os.remove(old_path)

def run_benchmark():
    start_time = datetime.now()
    results_summary = {}

    # Ensure the reports directory exists
    report_dir = os.path.join(current_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)

    for ticker in STOCKS:
        print(f"\n🚀 Processing Pipeline for: {ticker}...")
        stock_success = True
        
        # Pipeline Execution Steps
        steps = [
            ('Data_Ingestion', '1_get_real_data.py'),
            ('Feature_Engineering', '3_feature_engineering.py'),
            ('Model_Training', '4_train_model.py')
        ]
        
        for step_label, script_name in steps:
            script_path = os.path.join(current_dir, 'src', script_name)
            expected_csv = os.path.join(current_dir, 'data', f"{ticker.lower()}_assets.csv")
            had_file_before = False
            before_mtime = None
            if step_label == 'Data_Ingestion':
                had_file_before = os.path.exists(expected_csv)
                before_mtime = os.path.getmtime(expected_csv) if had_file_before else None
            
            try:
                # Execute script and capture errors
                print(f"📦 Executing {step_label}...")
                step_result = subprocess.run(
                    [sys.executable, script_path, ticker],
                    capture_output=True,
                    text=True
                )
                if step_result.stdout:
                    print(step_result.stdout, end="")
                if step_result.stderr:
                    print(step_result.stderr, end="")
                if step_result.returncode != 0:
                    raise RuntimeError(f"exit_code={step_result.returncode}")
                
                # Integrity Check: Verify if data was actually generated after ingestion
                if step_label == 'Data_Ingestion':
                    no_market_data = "No market data found" in (step_result.stdout + step_result.stderr)
                    has_file_after = os.path.exists(expected_csv)
                    after_mtime = os.path.getmtime(expected_csv) if has_file_after else None

                    file_not_updated = had_file_before and before_mtime == after_mtime
                    if no_market_data or (not has_file_after) or file_not_updated:
                        print(f"❌ Critical Error: {ticker} ingestion did not produce fresh raw data.")
                        stock_success = False
                        break
            except Exception as e:
                print(f"❌ {ticker} pipeline failed during {step_label}: {e}")
                stock_success = False
                break
        
        results_summary[ticker] = "Success" if stock_success else "Failed (Data/Execution Error)"

    # Persistence: Save the final audit report with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"performance_report_{timestamp}.json"
    report_path = os.path.join(report_dir, report_filename)

    final_report = {
        "timestamp": timestamp,
        "results": results_summary,
        "system_info": "MacBook Pro M4",
        "status": "Pipeline Run Complete"
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)
    prune_reports(report_dir, keep_latest=10)

    print(f"\n📊 Benchmark Audit Report saved to: {report_path}")
    print(f"📈 Execution Summary: {results_summary}")

if __name__ == "__main__":
    run_benchmark()