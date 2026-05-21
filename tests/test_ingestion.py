import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_ticker_exits_nonzero():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "1_get_real_data.py"), "NOTATICKER"],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "configs.json" in result.stdout + result.stderr
