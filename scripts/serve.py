#!/usr/bin/env python3
"""Minimal local inference API (FastAPI)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError:
    print("❌ Install serve extras: pip install '.[serve]'")
    sys.exit(1)

from pipeline.inference import predict_latest, train_and_save  # noqa: E402

app = FastAPI(
    title="Stationarity-Aware Market Modeling",
    description="Local inference API for saved tabular models.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train/{ticker}")
def train(ticker: str):
    try:
        path, meta = train_and_save(ticker.upper())
        return {"model_path": path, "meta": meta}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        return predict_latest(ticker.upper())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main():
    host = os.environ.get("SAMMM_HOST", "127.0.0.1")
    port = int(os.environ.get("SAMMM_PORT", "8000"))
    print(f"🚀 Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
