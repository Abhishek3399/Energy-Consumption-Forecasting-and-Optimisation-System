from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from .base import EnergyForecastModel


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        horizon: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_proj(x)
        enc = self.encoder(x)
        last = enc[:, -1, :]
        return self.fc(last)


class TransformerForecaster(EnergyForecastModel):
    name = "transformer"

    def __init__(
        self,
        input_size: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.horizon = horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = _TransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            horizon=horizon,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:  # type: ignore[override]
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x).float(), torch.from_numpy(y).float()
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            epoch_loss /= len(dataset)
            print(f"[Transformer] Epoch {epoch+1}/{self.epochs} - loss={epoch_loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(x).float().to(self.device)
            preds = self.model(inp).cpu().numpy()
        return preds

    def save(self, path: Path) -> None:  # type: ignore[override]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "meta": {
                    "input_size": self.input_size,
                    "horizon": self.horizon,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )

    def load(self, path: Path) -> None:  # type: ignore[override]
        ckpt = torch.load(path, map_location=self.device)
        meta = ckpt["meta"]
        self.__init__(
            input_size=meta["input_size"],
            horizon=meta["horizon"],
            d_model=meta["d_model"],
            nhead=meta["nhead"],
            num_layers=meta["num_layers"],
            lr=meta["lr"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            device=self.device,
        )
        self.model.load_state_dict(ckpt["state_dict"])

    def get_params(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "input_size": self.input_size,
            "horizon": self.horizon,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
        }

