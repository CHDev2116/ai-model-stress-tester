"""Model factory for walk-forward benchmarks."""

from sklearn.ensemble import RandomForestRegressor

from pipeline.training import REGRESSOR_PARAMS


def create_regressor(model_name="random_forest", **overrides):
    name = model_name.lower().replace("-", "_")
    if name in ("random_forest", "rf"):
        params = {**REGRESSOR_PARAMS, **overrides}
        return RandomForestRegressor(**params)
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("Install optional ML deps: pip install '.[ml]'") from exc
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            **overrides,
        }
        return xgb.XGBRegressor(**params)
    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install optional ML deps: pip install '.[ml]'") from exc
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            **overrides,
        }
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Unknown model: {model_name}")
