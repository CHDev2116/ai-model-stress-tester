# AI Model Stress Tester

A practical time-series ML pipeline for stress-testing stock prediction setups under high volatility.

This project focuses on **data stationarity**, **feature robustness**, and **quick model diagnostics** so you can rapidly compare target definitions (`Price`, `Simple_Return`, `Log_Return`) and understand model behavior.

## What This Project Does

- Downloads market data from Yahoo Finance
- Builds engineered features per ticker using `configs.json`
- Trains a `RandomForestRegressor` with a time-ordered split
- Reports MAE and saves a prediction chart to `reports/`

## Project Structure

```text
.
├── src/
│   ├── 1_get_real_data.py
│   ├── 2_explore_data.py
│   ├── 3_feature_engineering.py
│   ├── 4_train_model.py
│   └── configs.json
├── data/
├── reports/
├── benchmark.py
└── README.md
```

## Pipeline

1. **Data Ingestion** - `src/1_get_real_data.py`  
   Downloads data since `2023-01-01` to today and stores core columns (`Date`, `Close`, `Volume`) plus optional `High/Low` when available.

2. **Data Exploration** - `src/2_explore_data.py`  
   Quick preview + descriptive statistics for sanity checks.

3. **Feature Engineering** - `src/3_feature_engineering.py`  
   Generates moving averages, RSI, volatility variants, return targets, and saves processed dataset.

4. **Training & Profiling** - `src/4_train_model.py`  
   Trains Random Forest, computes MAE, prints latency/memory delta, and exports chart.

## Requirements

- Python 3.10+ (tested on Python 3.13)
- Packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `yfinance`
  - `psutil`

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib yfinance psutil
```

## Quick Start

From repo root:

```bash
# 1) Download raw data
python src/1_get_real_data.py GOOG

# 2) Optional profiling
python src/2_explore_data.py GOOG

# 3) Build features
python src/3_feature_engineering.py GOOG

# 4) Train and generate chart
python src/4_train_model.py GOOG
```

Output artifacts:

- Processed data: path from `src/configs.json -> <TICKER>.processed_output`
- Chart: `reports/chart_<TICKER>_<timestamp>.png`

## Configuration (`src/configs.json`)

Each ticker uses:

- `data_source`: raw CSV path
- `processed_output`: engineered CSV path
- `features`: feature columns used for training
- `target_type`: one of:
  - `Price`
  - `Simple_Return`
  - `Log_Return`

### Important Notes

- Training now warns if configured features are missing in processed data.
- If no valid features remain, training exits with an explicit error.
- `Price_Range` requires `High/Low`; when absent, feature engineering falls back to `abs(pct_change(Close))` as a proxy.

## Common Commands

```bash
# Run full pipeline for NVDA
python src/1_get_real_data.py NVDA
python src/3_feature_engineering.py NVDA
python src/4_train_model.py NVDA
```

## Troubleshooting

### 1) Terminal opens but wrong Python (`pyenv shims`)

Check:

```bash
which python
```

Expected:

```text
.../ai-model-stress-tester/.venv/bin/python
```

If not, run:

```bash
source .venv/bin/activate
```

### 2) Terminal in Cursor fails to open

- Reload window: `Developer: Reload Window`
- If still broken, fully quit Cursor (`Cmd+Q`) and reopen workspace
- Workspace terminal session corruption can be fixed by clearing workspace session cache (already addressed in this project setup)

### 3) Matplotlib cache/font warnings

If you see cache write warnings, set a writable config directory:

```bash
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
```

## License

For personal/portfolio use unless otherwise specified by repository owner.

## Baseline v1 (Current Stable Setup)

Use this as the default reference configuration before further tuning.

- **Training strategy**
  - Walk-forward inference (`retrain_every=20`)
  - RandomForest regressor + direction classifier blend
  - Recent drift anchoring for return targets
- **Model defaults**
  - Regressor: `n_estimators=300`, `max_depth=12`, `max_features='sqrt'`
  - Direction classifier: `n_estimators=200`, `max_depth=10`, `min_samples_leaf=2`
- **Target design**
  - `AVGO`: `Log_Return`
  - `GOOG`: `Log_Return`
  - `MU`: `Log_Return` with `lookback_days=730` (focus recent regime)
  - `NVDA`: `Simple_Return`
- **Reporting**
  - Return signal chart + reconstructed price path chart
  - `reports/` auto-prunes to keep latest 10 files

### Stability Checklist

Before changing hyperparameters again, run 3-5 benchmark rounds on different days and compare:

- MAE (% for return targets)
- Directional accuracy
- Price-path trend alignment (no persistent counter-trend behavior)
