"""Simple volatility-based regime labeling and per-regime metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error

from pipeline.training import TRAIN_FRACTION, predict_targets, prepare_xy


def label_volatility_regimes(df, vol_column="Volatility_20", quantile=0.5):
    """Label rows as high_vol / low_vol from rolling volatility."""
    if vol_column not in df.columns:
        temp_ret = np.log(df["Close"] / df["Close"].shift(1))
        vol = temp_ret.rolling(window=20).std()
    else:
        vol = df[vol_column]
    threshold = vol.quantile(quantile)
    regimes = np.where(vol >= threshold, "high_vol", "low_vol")
    return regimes, float(threshold)


def evaluate_by_regime(df, config, target_type=None):
    target_type = target_type or config["target_type"]
    regimes, threshold = label_volatility_regimes(df)
    df = df.copy()
    df["_regime"] = regimes

    X, y, features = prepare_xy(df, config)
    split = int(len(df) * TRAIN_FRACTION)
    preds = predict_targets(X, y, split, target_type)

    results = {"vol_threshold": threshold, "regimes": {}, "features": features}
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[split:] = True

    for regime in ("low_vol", "high_vol"):
        mask = test_mask & (df["_regime"].values == regime)
        if not mask.any():
            continue
        idx = np.where(mask)[0] - split
        y_true = y.iloc[split:].values[idx]
        y_pred = preds[idx]
        mae = float(mean_absolute_error(y_true, y_pred))
        entry = {"n_test": int(mask.sum()), "mae": mae}
        if "Return" in target_type:
            entry["mae_pct"] = mae * 100
            entry["direction_accuracy"] = float(
                np.mean(np.sign(y_true) == np.sign(y_pred))
            )
        results["regimes"][regime] = entry

    return results
