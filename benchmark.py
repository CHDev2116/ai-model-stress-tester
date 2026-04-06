import subprocess
import os
import json
from datetime import datetime

# Configuration
STOCKS = ['AVGO', 'GOOG', 'MU', 'NVDA']
current_dir = os.path.dirname(os.path.abspath(__file__))

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
            
            try:
                # Execute script and capture errors
                print(f"📦 Executing {step_label}...")
                subprocess.run(['python3', script_path, ticker], check=True)
                
                # Integrity Check: Verify if data was actually generated after ingestion
                if step_label == 'Data_Ingestion':
                    expected_csv = os.path.join(current_dir, 'data', f"{ticker.lower()}_assets.csv")
                    if not os.path.exists(expected_csv):
                        print(f"❌ Critical Error: Data artifact {expected_csv} is missing!")
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

    print(f"\n📊 Benchmark Audit Report saved to: {report_path}")
    print(f"📈 Execution Summary: {results_summary}")

if __name__ == "__main__":
    run_benchmark()