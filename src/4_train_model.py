import sys

from pipeline.config import fail
from pipeline.training import run_training

if __name__ == "__main__":
    target_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    try:
        run_training(target_ticker, write_charts=True)
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"❌ Training failed for {target_ticker}: {exc}")
    sys.exit(0)
