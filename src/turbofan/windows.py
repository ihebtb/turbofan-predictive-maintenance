"""Sliding windows over each engine's history, and tabular features from them."""

import numpy as np
import pandas as pd


def _padded(values: np.ndarray, length: int) -> np.ndarray:
    """Left-pad a short history by repeating its first row."""
    if len(values) >= length:
        return values
    pad = np.repeat(values[:1], length - len(values), axis=0)
    return np.vstack([pad, values])


def training_windows(df: pd.DataFrame, sensors: list[str], length: int = 30):
    """Every window of `length` cycles, labelled with the RUL at its last cycle."""
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        values = _padded(g[sensors].to_numpy(dtype=np.float32), length)
        rul = g["rul"].to_numpy(dtype=np.float32)
        rul = np.concatenate([np.full(len(values) - len(rul), rul[0]), rul])
        for end in range(length, len(values) + 1):
            xs.append(values[end - length:end])
            ys.append(rul[end - 1])
            units.append(unit)
    return np.stack(xs), np.array(ys, dtype=np.float32), np.array(units)


def last_windows(df: pd.DataFrame, sensors: list[str], length: int = 30):
    """The most recent window of each engine (what is known at prediction time)."""
    xs, units = [], []
    for unit, g in df.groupby("unit", sort=True):
        values = _padded(g[sensors].to_numpy(dtype=np.float32), length)
        xs.append(values[-length:])
        units.append(unit)
    return np.stack(xs), np.array(units)


def window_features(x: np.ndarray) -> np.ndarray:
    """Per sensor: last value, mean, standard deviation and linear trend of the window."""
    n, length, _ = x.shape
    t = np.arange(length, dtype=np.float32)
    t = (t - t.mean()) / (t.std() or 1.0)
    slope = np.einsum("t,nts->ns", t, x - x.mean(axis=1, keepdims=True)) / length
    return np.concatenate([x[:, -1, :], x.mean(axis=1), x.std(axis=1), slope], axis=1)
