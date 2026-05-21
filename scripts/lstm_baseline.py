#!/usr/bin/env python3
"""
Minimal LSTM baseline on return targets (single chronological 80/20 split).

Requires: pip install ".[torch]"
Compare with tabular RF via scripts/sequence_baseline.py or scripts/benchmark_models.py.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline.config import fail, load_config  # noqa: E402
from pipeline.models import create_regressor  # noqa: E402
from pipeline.paths import resolve_path  # noqa: E402
from pipeline.training import TRAIN_FRACTION, load_training_frame, walk_forward_predictions  # noqa: E402

LOOKBACK = 20
EPOCHS = 25
BATCH_SIZE = 32
HIDDEN_SIZE = 32
LR = 1e-3
SEED = 42


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        fail('❌ PyTorch not installed. Run: pip install ".[torch]"')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return torch, nn, DataLoader, TensorDataset


def build_sequences(X_scaled, y, lookback):
    """Rows i use features [i-lookback:i) to predict y[i]."""
    rows, n_feat = X_scaled.shape
    seq_X, seq_y, indices = [], [], []
    for i in range(lookback, rows):
        seq_X.append(X_scaled[i - lookback : i])
        seq_y.append(y.iloc[i])
        indices.append(i)
    return np.array(seq_X, dtype=np.float32), np.array(seq_y, dtype=np.float32), indices


class LSTMRegressor:
    """Thin wrapper so training loop stays readable."""

    def __init__(self, n_features, hidden, nn_module, torch_mod):
        self.torch = torch_mod
        self.net = nn_module.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )
        self.head = nn_module.Linear(hidden, 1)
        self.hidden = hidden
        self.n_features = n_features

    def to(self, device):
        self.net.to(device)
        self.head.to(device)
        return self

    def forward(self, x):
        out, _ = self.net(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_lstm(seq_X_train, seq_y_train, n_features, device, torch_mod, nn_module, DataLoader):
    model = LSTMRegressor(n_features, HIDDEN_SIZE, nn_module, torch_mod).to(device)
    optimizer = torch_mod.optim.Adam(
        list(model.net.parameters()) + list(model.head.parameters()),
        lr=LR,
    )
    loss_fn = nn_module.MSELoss()

    X_t = torch_mod.tensor(seq_X_train, device=device)
    y_t = torch_mod.tensor(seq_y_train, device=device)
    loader = DataLoader(
        torch_mod.utils.data.TensorDataset(X_t, y_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.net.train()
    model.head.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model.forward(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model


def predict_lstm(model, seq_X, device, torch_mod):
    model.net.eval()
    model.head.eval()
    with torch_mod.no_grad():
        preds = model.forward(torch_mod.tensor(seq_X, device=device))
    return preds.cpu().numpy()


def evaluate_lstm(df, config):
    torch_mod, nn_module, DataLoader, _ = _require_torch()

    features = [f for f in config["features"] if f in df.columns]
    X, y = df[features], df["Target"]
    split = int(len(df) * TRAIN_FRACTION)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    seq_X, seq_y, indices = build_sequences(X_scaled, y, LOOKBACK)
    indices = np.array(indices)

    train_mask = indices < split
    test_mask = indices >= split
    if train_mask.sum() < 50 or test_mask.sum() < 10:
        fail("❌ Not enough sequences after lookback for LSTM train/test split.")

    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    model = train_lstm(
        seq_X[train_mask],
        seq_y[train_mask],
        len(features),
        device,
        torch_mod,
        nn_module,
        DataLoader,
    )
    preds = predict_lstm(model, seq_X[test_mask], device, torch_mod)
    y_test = seq_y[test_mask]
    mae = float(mean_absolute_error(y_test, preds))

    result = {
        "model": "lstm",
        "lookback": LOOKBACK,
        "epochs": EPOCHS,
        "hidden_size": HIDDEN_SIZE,
        "device": str(device),
        "n_train_seq": int(train_mask.sum()),
        "n_test_seq": int(test_mask.sum()),
        "mae": mae,
    }
    if "Return" in config["target_type"]:
        result["mae_pct"] = round(mae * 100, 4)
        result["direction_accuracy"] = float(
            np.mean(np.sign(y_test) == np.sign(preds))
        )
    return result


def evaluate_rf_same_split(df, config):
    features = [f for f in config["features"] if f in df.columns]
    X, y = df[features], df["Target"]
    split = int(len(df) * TRAIN_FRACTION)
    preds = walk_forward_predictions(
        X, y, split, model_factory=lambda: create_regressor("rf")
    )
    y_test = y.iloc[split:]
    mae = float(mean_absolute_error(y_test, preds))
    out = {"model": "random_forest_walk_forward", "mae": mae}
    if "Return" in config["target_type"]:
        out["mae_pct"] = round(mae * 100, 4)
        out["direction_accuracy"] = float(
            np.mean(np.sign(y_test.values) == np.sign(preds))
        )
    return out


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOG"
    config = load_config(ticker)
    df = load_training_frame(ticker, config)

    print(f"🧠 LSTM baseline for {ticker} (lookback={LOOKBACK}, epochs={EPOCHS})...")
    lstm_metrics = evaluate_lstm(df, config)
    rf_metrics = evaluate_rf_same_split(df, config)

    report = {
        "ticker": ticker,
        "target_type": config["target_type"],
        "protocol": "80/20 chronological split; LSTM single fit on train sequences",
        "lstm": lstm_metrics,
        "random_forest": rf_metrics,
    }
    print(json.dumps(report, indent=2))

    out = resolve_path(
        f"reports/lstm_baseline_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"📄 Saved: {out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
