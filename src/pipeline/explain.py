"""SHAP-based feature importance (optional dependency)."""

import json
import os
from datetime import datetime

from pipeline.paths import resolve_path
from pipeline.training import REGRESSOR_PARAMS, prepare_xy


def compute_shap_summary(df, config, max_samples=200):
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise ImportError("Install optional ML deps: pip install '.[ml]'") from exc

    X, y, features = prepare_xy(df, config)
    split = int(len(df) * 0.8)
    model = RandomForestRegressor(**REGRESSOR_PARAMS)
    model.fit(X.iloc[:split], y.iloc[:split])

    sample = X.iloc[split : split + max_samples]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    mean_abs = abs(values).mean(axis=0)
    ranking = sorted(
        zip(features, mean_abs.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "features": features,
        "mean_abs_shap": {name: round(val, 6) for name, val in ranking},
        "top_features": [name for name, _ in ranking[:5]],
    }


def save_shap_report(ticker, df, config, report_dir=None):
    report_dir = report_dir or resolve_path("reports")
    os.makedirs(report_dir, exist_ok=True)
    summary = compute_shap_summary(df, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(report_dir, f"shap_{ticker.upper()}_{timestamp}.json")
    payload = {"ticker": ticker.upper(), "timestamp": timestamp, **summary}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path, payload
