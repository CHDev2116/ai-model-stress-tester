# AI Model Stress Tester

Stress-test framework for stock time-series modeling under high volatility.

This project demonstrates how target design and feature engineering can dramatically improve model stability when prices trend strongly.

## Highlights

- Built a modular 4-stage ML workflow: ingestion -> diagnostics -> feature engineering -> training/profiling
- Improved prediction behavior by shifting from absolute price targets to return-based targets
- Added pipeline safeguards for feature mismatch and missing column scenarios
- Produces reproducible artifacts (`data/*.csv`, `reports/chart_*.png`) for model comparison

## Why This Project

Traditional regressors on raw prices often degrade when asset scales drift over time (e.g., large rallies).  
This repo focuses on **stationarity-aware modeling**:

- `Price` target for baseline
- `Simple_Return` and `Log_Return` for scale-invariant targets
- Technical and volatility features to better capture regime shifts

## Tech Stack

- Python (`pandas`, `numpy`, `scikit-learn`, `yfinance`, `matplotlib`, `psutil`)
- Model: `RandomForestRegressor`
- Evaluation: Time-ordered split + MAE + visual signal comparison

## Demo (Run in 1 minute)

```bash
python src/1_get_real_data.py GOOG
python src/3_feature_engineering.py GOOG
python src/4_train_model.py GOOG
```

After running:

- Processed dataset is written to the ticker path defined in `src/configs.json`
- Performance chart is saved to `reports/`

## Repository Map

```text
src/1_get_real_data.py        # download and clean market data
src/2_explore_data.py         # quick dataset profiling
src/3_feature_engineering.py  # indicators + target construction
src/4_train_model.py          # training, MAE, profiling, chart export
src/configs.json              # per-ticker feature/target/output config
```

## Engineering Notes

- Feature engineering handles missing `High/Low` gracefully for `Price_Range` by using a fallback proxy.
- Training warns when configured features are missing and fails fast if none are usable.
- Data ingestion uses dynamic end date (today) instead of a hardcoded cutoff.

## Portfolio Context

This project is suitable as an ML performance engineering case study:

- **Problem framing**: drift and extrapolation failure in non-stationary financial series
- **Method**: target redesign + feature re-engineering + controlled profiling
- **Outcome**: improved robustness and clearer diagnostics for iterative modeling

## Developer Docs

Detailed setup, troubleshooting, and operational commands are in `README.dev.md`.