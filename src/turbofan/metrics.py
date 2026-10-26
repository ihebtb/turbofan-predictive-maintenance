import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 challenge score (lower is better). Late predictions (overestimating the
    remaining life) are penalized more than early ones, because they risk a failure
    in service: exp(-d/13) - 1 if early, exp(d/10) - 1 if late, with d = pred - true."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
