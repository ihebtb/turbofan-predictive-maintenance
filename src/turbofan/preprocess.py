"""Sensor selection and operating-condition-aware normalization."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import SENSORS, SETTINGS


@dataclass
class Normalizer:
    """Standardize sensors per operating condition.

    FD002 and FD004 mix six flight conditions (altitude, Mach, throttle). The
    same healthy engine reads very differently in each, so sensors are
    standardized within each condition, found by clustering the three setting
    columns. With a single condition this reduces to a global z-score.
    Sensors that barely vary inside every condition carry no degradation signal
    and are dropped.
    """

    n_conditions: int = 1
    min_std: float = 1e-4
    sensors: list[str] = field(default_factory=list)
    _kmeans: KMeans | None = None
    _stats: dict = field(default_factory=dict)

    def _conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self._kmeans is None:
            return np.zeros(len(df), dtype=int)
        return self._kmeans.predict(df[SETTINGS].round(2).to_numpy())

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        if self.n_conditions > 1:
            self._kmeans = KMeans(self.n_conditions, n_init=10, random_state=0)
            self._kmeans.fit(df[SETTINGS].round(2).to_numpy())
        cond = self._conditions(df)
        self._stats = {}
        stds = []
        for c in np.unique(cond):
            part = df.loc[cond == c, SENSORS]
            self._stats[int(c)] = (part.mean(), part.std(ddof=0))
            stds.append(part.std(ddof=0))
        typical = pd.concat(stds, axis=1).max(axis=1)
        means = pd.concat([m for m, _ in self._stats.values()], axis=1).abs().max(axis=1)
        relative = typical / means.replace(0, np.nan)
        self.sensors = [s for s in SENSORS if relative.get(s, 0) > self.min_std]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cond = self._conditions(df)
        for c, (mean, std) in self._stats.items():
            mask = cond == c
            out.loc[mask, self.sensors] = (
                (df.loc[mask, self.sensors] - mean[self.sensors]) / std[self.sensors].replace(0, 1)
            ).to_numpy()
        out["condition"] = cond
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def conditions_for(subset: str) -> int:
    return 6 if subset in ("FD002", "FD004") else 1
