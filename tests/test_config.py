import pytest

from pipeline.config import load_config, load_all_configs


def test_load_all_configs_has_benchmark_tickers():
    configs = load_all_configs()
    assert set(configs.keys()) == {"AVGO", "GOOG", "MU", "NVDA"}


def test_load_config_goog():
    config = load_config("goog")
    assert config["target_type"] == "Log_Return"
    assert "MA5" in config["features"]


def test_load_config_missing_ticker():
    with pytest.raises(KeyError):
        load_config("FAKE")
