from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.features import engineer_dataframe
from pipeline.training import run_training


def _resolve_in_tmp(tmp_path, relative):
    if relative == ".":
        return str(tmp_path)
    return str(tmp_path / relative)


def test_run_training_emits_metrics(tmp_path, synthetic_ohlcv, goog_config, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()

    config = {
        **goog_config,
        "data_source": "data/processed_goog.csv",
        "processed_output": "data/processed_goog.csv",
    }

    processed = engineer_dataframe(synthetic_ohlcv.copy(), config)
    processed.to_csv(tmp_path / "data" / "processed_goog.csv", index=False)

    monkeypatch.setattr(
        "pipeline.training.resolve_path",
        lambda rel: _resolve_in_tmp(tmp_path, rel),
    )
    monkeypatch.setattr(
        "pipeline.training.load_config",
        lambda ticker: config,
    )

    metrics = run_training("GOOG", write_charts=False)

    assert metrics["ticker"] == "GOOG"
    assert metrics["mae"] >= 0
    assert metrics["direction_accuracy"] is not None
    assert (tmp_path / "reports").exists()
    metrics_files = list((tmp_path / "reports").glob("metrics_GOOG_*.json"))
    assert len(metrics_files) == 1


def test_load_config_integrates_with_repo():
    config = load_config("GOOG")
    assert config["target_type"] in ("Log_Return", "Simple_Return", "Price")
