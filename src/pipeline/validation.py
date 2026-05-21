"""Rolling / expanding walk-forward validation."""

import numpy as np
from sklearn.metrics import mean_absolute_error

from pipeline.training import (
    MIN_SAMPLES,
    TRAIN_FRACTION,
    predict_targets,
    prepare_xy,
)


def rolling_walk_forward_metrics(df, config, target_type=None, n_folds=5):
    """
    Expanding-window folds on the test region (last 20% by default).
    Each fold trains on all data up to fold start and evaluates the next segment.
    """
    target_type = target_type or config["target_type"]
    X, y, features = prepare_xy(df, config)
    split = int(len(df) * TRAIN_FRACTION)
    test_len = len(df) - split
    if test_len < n_folds * 5:
        n_folds = max(1, test_len // 5)

    preds_full = predict_targets(X, y, split, target_type)
    fold_size = max(1, test_len // n_folds)
    folds = []

    for fold_idx in range(n_folds):
        test_start = split + fold_idx * fold_size
        test_end = split + (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else len(df)
        if test_start >= len(df) or test_end <= test_start:
            continue

        y_true = y.iloc[test_start:test_end].values
        preds_seg = preds_full[test_start - split : test_end - split]

        mae = float(mean_absolute_error(y_true, preds_seg))
        direction_acc = None
        if "Return" in target_type:
            direction_acc = float(np.mean(np.sign(y_true) == np.sign(preds_seg)))

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_end_idx": split,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
                "n_test": len(y_true),
                "mae": mae,
                "mae_pct": mae * 100 if "Return" in target_type else None,
                "direction_accuracy": direction_acc,
            }
        )

    maes = [f["mae"] for f in folds]
    summary = {
        "target_type": target_type,
        "n_folds": len(folds),
        "features": features,
        "mae_mean": float(np.mean(maes)) if maes else None,
        "mae_std": float(np.std(maes)) if maes else None,
        "folds": folds,
    }
    if folds and folds[0].get("direction_accuracy") is not None:
        dirs = [f["direction_accuracy"] for f in folds if f["direction_accuracy"] is not None]
        summary["direction_accuracy_mean"] = float(np.mean(dirs))
    return summary
