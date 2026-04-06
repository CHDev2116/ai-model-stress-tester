# Financial-ML-Performance-Optimizer
**A specialized framework for diagnosing and optimizing ML model performance on high-volatility time-series data.**

## 🚀 Overview
This project demonstrates a professional ML performance engineering workflow. Originally, standard regression models faced significant "Scale-Drift" issues (MAE > 125) when predicting high-growth stocks like MU and GOOG (which rallied from 180 to 320). 

By implementing **Log-Return Stationarity** and **Feature Re-engineering**, this project successfully optimized model precision, reducing Mean Absolute Error (MAE) from **31.27** to **1.42%**.

## 🎯 Key Achievements (Targeting Waymo ML Performance)
- **Error Reduction**: Reduced GOOG prediction MAE by **95.4%** through data stationarity transformation.
- **Diagnostic Tooling**: Built a decoupled 2-stage pipeline (Feature Engineering & Model Profiling) to isolate data-drift issues.
- **Scale Invariance**: Solved the "Extrapolation Failure" in Random Forest models by switching from absolute price targets to log-return targets.

## 🛠️ System Architecture
The project is structured into modular components to ensure rapid iteration and performance monitoring:

1. **`1_get_real_data.py`**: Automated data ingestion from Yahoo Finance.
2. **`3_feature_engineering.py`**: 
    - Implements Log-Return calculation.
    - Features: RSI, Bias (Mean Reversion), MA-Ratio (Trend), and Rolling Volatility.
    - **Performance Logic**: Converts non-stationary price data into stationary return distributions.
3. **`4_train_model.py`**:
    - Machine Learning Profiler using Random Forest Regressor.
    - Automated evaluation of MAE and visual trend alignment.

## 📊 Performance Metrics (Case Study: GOOG)
| Phase | Metric (MAE) | Logic Applied | Result |
| :--- | :--- | :--- | :--- |
| **Initial** | 31.27 | Raw Price Prediction | Failure (Scale-Drift) |
| **Optimized** | **1.42%** | Log-Return Prediction | **Success (Trend-Aligned)** |



## 💻 Environment & Requirements
- **Hardware**: Optimized for Apple M4 Silicon (Parallel task execution).
- **Stack**: Python 3.13, Pandas, Scikit-learn, Matplotlib.
- **Performance Profiling**: Included latency and memory tracking for local LLM orchestration (Gemma 4).

## 📝 How to Run
1. Ensure your environment has `pandas`, `scikit-learn`, and `yfinance`.
2. Run data engineering: `python 3_feature_engineering.py`
3. Execute performance profiling: `python 4_train_model.py`

---
**Contact**: Developed by Cheryl - AI Quality Assurance & Performance Engineer.