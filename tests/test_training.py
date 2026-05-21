import numpy as np

from pipeline.features import engineer_dataframe
from pipeline.training import (
    evaluate_dataframe,
    predict_targets,
    walk_forward_predictions,
)


def test_walk_forward_predictions_length(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    features = [f for f in goog_config["features"] if f in df.columns]
    X, y = df[features], df["Target"]
    split = int(len(df) * 0.8)
    preds = walk_forward_predictions(X, y, split)
    assert len(preds) == len(y) - split


def test_evaluate_dataframe_metrics(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    metrics = evaluate_dataframe(df, goog_config)
    assert metrics["target_type"] == "Log_Return"
    assert metrics["mae"] >= 0
    assert metrics["direction_accuracy"] is not None
    assert 0 <= metrics["direction_accuracy"] <= 1


def test_price_target_mae_differs_from_return(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    return_metrics = evaluate_dataframe(df, goog_config, target_type="Log_Return")
    price_metrics = evaluate_dataframe(df, goog_config, target_type="Price")
    assert return_metrics["mae"] != price_metrics["mae"]
