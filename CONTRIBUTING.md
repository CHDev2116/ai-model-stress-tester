# Contributing

Thanks for improving **Stationarity-Aware Market Modeling**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
```

Optional stacks: `pip install ".[ml]"`, `pip install ".[torch]"`, `pip install ".[serve]"`.

## Checks before a PR

```bash
# Fast tests (excludes slow LSTM training)
pytest -q -m "not slow"

# Offline pipeline smoke (synthetic CSVs)
mkdir -p data reports
# … see .github/workflows/ci.yml for the seed-data snippet …
PYTHONPATH=src python src/3_feature_engineering.py GOOG
PYTHONPATH=src python src/4_train_model.py GOOG

# Refresh README metrics after real data runs
python scripts/refresh_readme_metrics.py
```

## Release / publish checklist

1. `rm -rf build dist *.egg-info src/*.egg-info`
2. Ensure committed PNGs exist: `reports/chart_GOOG.png`, `reports/chart_GOOG_pricepath.png`, `reports/chart_NVDA_failure.png`
3. Run `python benchmark.py` (or `--skip-ingestion` with fresh local CSVs)
4. Run `python scripts/refresh_readme_metrics.py`
5. `git status` — commit pipeline, scripts, tests, CI, docs, and stable report images
6. Optional: rename repo per [`docs/GITHUB_REPO.md`](docs/GITHUB_REPO.md)

## Code layout

- Put shared logic in `src/pipeline/`, not duplicated in `scripts/`.
- CLI wrappers stay in `src/1_*.py` and `scripts/`.
- Failures should call `pipeline.config.fail()` so exit code is non-zero.

## License

Contributions are accepted under the [MIT License](LICENSE).
