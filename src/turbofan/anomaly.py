"""Unsupervised health monitoring: learn what a healthy engine looks like, then flag
cycles that drift away from it, and say which sensors explain the drift.

Every C-MAPSS engine starts healthy, so the first cycles of each training engine
are used as the "healthy" reference. No failure label is used to fit the detector.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .data import SENSOR_INFO


@dataclass
class HealthMonitor:
    healthy_cycles: int = 30
    quantile: float = 0.995
    consecutive: int = 3
    smooth: int = 5
    sensors: list[str] = field(default_factory=list)
    threshold: float = 0.0
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None
    _precision: np.ndarray | None = None

    def _healthy(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["cycle"] <= self.healthy_cycles]

    def fit(self, df: pd.DataFrame, sensors: list[str], calibration: pd.DataFrame | None = None):
        """Fit on early-life cycles of `df`; set the alarm threshold on the early-life
        cycles of `calibration` (other engines), so the threshold is not tuned on the
        very data the covariance was fitted on."""
        self.sensors = list(sensors)
        healthy = self._healthy(df)[self.sensors].to_numpy()
        self._mean, self._std = healthy.mean(0), healthy.std(0) + 1e-6
        cov = LedoitWolf().fit((healthy - self._mean) / self._std)
        self._precision = cov.precision_
        ref = self._healthy(calibration if calibration is not None else df)
        self.threshold = float(np.quantile(self.score_frame(ref), self.quantile))
        return self

    def _z(self, values: np.ndarray) -> np.ndarray:
        return (values - self._mean) / self._std

    def raw_score(self, values: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance to the healthy state, per cycle."""
        z = self._z(values)
        return np.einsum("ij,jk,ik->i", z, self._precision, z)

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Per-cycle score, smoothed with a short rolling mean within each engine."""
        scores = pd.Series(self.raw_score(df[self.sensors].to_numpy()), index=df.index)
        return scores.groupby(df["unit"]).transform(
            lambda s: s.rolling(self.smooth, min_periods=1).mean()
        ).to_numpy()

    def alarms(self, df: pd.DataFrame) -> pd.Series:
        """True from the first cycle where the score stayed above threshold for
        `consecutive` cycles in a row."""
        above = pd.Series(self.score_frame(df) > self.threshold, index=df.index)
        run = above.groupby(df["unit"]).transform(
            lambda s: s.rolling(self.consecutive, min_periods=self.consecutive).sum()
        )
        return (run >= self.consecutive).fillna(False)

    def contributions(self, values: np.ndarray) -> pd.DataFrame:
        """Share of the anomaly score explained by each sensor (z * (P z), summed = score)."""
        z = self._z(np.atleast_2d(values))
        parts = z * (z @ self._precision)
        total = parts.sum(axis=1, keepdims=True)
        share = np.where(total > 0, parts / np.where(total == 0, 1, total), 0)
        return pd.DataFrame(share, columns=self.sensors)

    def explain(self, values: np.ndarray, top: int = 3) -> list[dict]:
        """Top sensors behind the anomaly at one cycle, with direction and physical name."""
        share = self.contributions(values).iloc[0]
        z = self._z(np.atleast_2d(values))[0]
        out = []
        for sensor in share.sort_values(ascending=False).index[:top]:
            symbol, description = SENSOR_INFO[sensor]
            i = self.sensors.index(sensor)
            out.append({
                "sensor": sensor, "symbol": symbol, "description": description,
                "share": round(float(share[sensor]), 3),
                "direction": "above healthy range" if z[i] > 0 else "below healthy range",
                "z": round(float(z[i]), 2),
            })
        return out


def detection_report(monitor: HealthMonitor, df: pd.DataFrame) -> dict:
    """On run-to-failure trajectories: how early does the alarm fire before failure,
    and how often does it fire while the engine is still in its healthy phase?"""
    alarm = monitor.alarms(df)
    lead_times, early_false = [], 0
    for _, g in df.groupby("unit"):
        fired = g.loc[alarm[g.index], "cycle"]
        last = g["cycle"].max()
        if not fired.empty:
            lead_times.append(int(last - fired.iloc[0]))
            if fired.iloc[0] <= monitor.healthy_cycles:
                early_false += 1
    n = df["unit"].nunique()
    return {
        "engines": int(n),
        "detected_before_failure": round(len(lead_times) / n, 3),
        "median_lead_time_cycles": float(np.median(lead_times)) if lead_times else None,
        "p10_lead_time_cycles": float(np.percentile(lead_times, 10)) if lead_times else None,
        "alarm_in_healthy_phase": round(early_false / n, 3),
    }


def inject_faults(df: pd.DataFrame, sensors: list[str], rate: float = 0.05,
                  seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject sensor faults into healthy cycles to test detection: spikes (one
    cycle, +/-6 sd), stuck-at values and a slow drift, on one random sensor each."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    healthy_idx = df.index[df["cycle"] <= 30].to_numpy()
    n = int(len(healthy_idx) * rate)
    chosen = rng.choice(healthy_idx, size=n, replace=False)
    labels = np.zeros(len(df), dtype=bool)
    pos = {idx: i for i, idx in enumerate(df.index)}
    std = df.loc[healthy_idx, sensors].std().to_numpy()
    for idx in chosen:
        j = rng.integers(len(sensors))
        sensor = sensors[j]
        kind = rng.choice(["spike", "stuck", "drift"])
        if kind == "spike":
            out.loc[idx, sensor] += rng.choice([-1, 1]) * 6 * std[j]
        elif kind == "stuck":
            out.loc[idx, sensor] = df[sensor].quantile(0.999) + 3 * std[j]
        else:
            out.loc[idx, sensor] += 5 * std[j]
        labels[pos[idx]] = True
    return out, labels
