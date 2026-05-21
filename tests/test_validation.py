from pipeline.features import engineer_dataframe
from pipeline.regime import evaluate_by_regime
from pipeline.validation import rolling_walk_forward_metrics


def test_rolling_walk_forward_folds(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    report = rolling_walk_forward_metrics(df, goog_config, n_folds=3)
    assert report["n_folds"] >= 1
    assert report["mae_mean"] is not None


def test_regime_evaluation(synthetic_ohlcv, goog_config):
    df = engineer_dataframe(synthetic_ohlcv, goog_config)
    report = evaluate_by_regime(df, goog_config)
    assert "regimes" in report
    assert len(report["regimes"]) >= 1
