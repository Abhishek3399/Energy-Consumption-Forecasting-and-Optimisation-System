from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader

from ..config import MODELS_DIR
from ..data.dataset import SlidingWindowDataset
from .evaluate import regression_metrics
from .lstm_model import LSTMForecaster
from .transformer_model import TransformerForecaster


ModelType = Literal["lstm", "transformer"]


def train_sequence_model(
    model_type: ModelType,
    train_ds: SlidingWindowDataset,
    val_ds: SlidingWindowDataset,
    input_size: int,
    horizon: int,
    model_name: str,
) -> dict:
    if model_type == "lstm":
        model = LSTMForecaster(input_size=input_size, horizon=horizon)
    else:
        model = TransformerForecaster(input_size=input_size, horizon=horizon)

    train_loader = DataLoader(train_ds, batch_size=model.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=model.batch_size, shuffle=False)

    # simple training loop delegating to model.fit for now
    x_train, y_train = next(iter(DataLoader(train_ds, batch_size=len(train_ds))))
    model.fit(x_train.numpy(), y_train.numpy())

    # evaluation
    xs_val, ys_val = [], []
    for xb, yb in val_loader:
        xs_val.append(xb.numpy())
        ys_val.append(yb.numpy())
    x_val_np = np.concatenate(xs_val, axis=0)
    y_val_np = np.concatenate(ys_val, axis=0)
    preds = model.predict(x_val_np)
    metrics = regression_metrics(y_val_np, preds)

    ckpt_path = MODELS_DIR / f"{model_name}_{model_type}.pt"
    model.save(ckpt_path)

    return {
        "model_type": model_type,
        "model_name": model_name,
        "metrics": metrics,
        "path": str(ckpt_path),
    }

