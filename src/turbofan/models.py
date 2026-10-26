"""Remaining-useful-life regressors: a linear baseline, gradient boosting and an LSTM."""

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch import nn

from .windows import window_features


class LinearBaseline:
    """Ridge regression on the sensors of the last cycle only (no history)."""

    name = "linear_last_cycle"

    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def fit(self, x: np.ndarray, y: np.ndarray, **_):
        self.model.fit(x[:, -1, :], y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(x[:, -1, :]), 0, None)


class GradientBoosting:
    """Histogram gradient boosting on window statistics (last, mean, std, trend)."""

    name = "gradient_boosting"

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=seed,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, **_):
        self.model.fit(window_features(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(window_features(x)), 0, None)


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMRegressor:
    """2-layer LSTM over the window, trained with early stopping on validation engines."""

    name = "lstm"

    def __init__(self, hidden: int = 64, epochs: int = 60, batch_size: int = 256,
                 lr: float = 1e-3, patience: int = 8, seed: int = 0):
        self.hidden, self.epochs, self.batch_size = hidden, epochs, batch_size
        self.lr, self.patience, self.seed = lr, patience, seed
        self.net: _LSTMNet | None = None
        self.history: list[dict] = []
        self.y_scale = 125.0

    def fit(self, x: np.ndarray, y: np.ndarray, x_val=None, y_val=None):
        torch.manual_seed(self.seed)
        self.y_scale = float(max(y.max(), 1.0))
        self.net = _LSTMNet(x.shape[2], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        xt, yt = torch.from_numpy(x), torch.from_numpy(y / self.y_scale)
        best, best_state, bad = np.inf, None, 0
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), self.batch_size):
                idx = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if x_val is None:
                continue
            val = float(np.sqrt(np.mean((self.predict(x_val) - y_val) ** 2)))
            self.history.append({"epoch": epoch + 1, "val_rmse": round(val, 3)})
            if val < best - 1e-3:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(x), 2048):
                out.append(self.net(torch.from_numpy(np.ascontiguousarray(x[i:i + 2048]))).numpy())
        return np.clip(np.concatenate(out) * self.y_scale, 0, None)

    def state(self) -> dict:
        return {"state_dict": self.net.state_dict(), "n_features": self.net.lstm.input_size,
                "hidden": self.hidden, "y_scale": self.y_scale}

    @classmethod
    def from_state(cls, state: dict) -> "LSTMRegressor":
        model = cls(hidden=state["hidden"])
        model.net = _LSTMNet(state["n_features"], state["hidden"])
        model.net.load_state_dict(state["state_dict"])
        model.y_scale = state["y_scale"]
        return model
