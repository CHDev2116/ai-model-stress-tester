import json
import os
import sys

from pipeline.paths import SRC_DIR

CONFIG_PATH = os.path.join(SRC_DIR, "configs.json")


def project_root():
    return os.path.dirname(SRC_DIR)


def src_dir():
    return SRC_DIR


def load_all_configs():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_config(ticker):
    configs = load_all_configs()
    key = ticker.upper()
    if key not in configs:
        raise KeyError(f"Configuration for {key} not found")
    return configs[key]


def fail(message, code=1):
    print(message)
    sys.exit(code)
