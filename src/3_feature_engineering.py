import sys

from pipeline.features import run_feature_engineering

if __name__ == "__main__":
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    run_feature_engineering(target_ticker)
    sys.exit(0)
