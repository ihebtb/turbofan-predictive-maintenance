"""Load trained artifacts and produce an engine assessment: remaining life, health
score, alarm state and the sensors that explain the anomaly."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import SENSORS, SETTINGS
from .models import LSTMRegressor
from .windows import last_windows


class EngineAssessor:
    def __init__(self, model_dir: Path | str, subset: str = "FD001"):
        base = Path(model_dir) / subset
        if not base.exists():
            raise FileNotFoundError(f"{base} not found. Train first: python -m turbofan.train")
        self.subset = subset
        self.meta = json.loads((base / "meta.json").read_text())
        self.normalizer = joblib.load(base / "normalizer.joblib")
        self.gbm = joblib.load(base / "gradient_boosting.joblib")
        self.lstm = LSTMRegressor.from_state(torch.load(base / "lstm.pt", weights_only=False))
        self.monitor = joblib.load(base / "health_monitor.joblib")

    def assess(self, history: pd.DataFrame) -> dict:
        """`history`: one engine's cycles (columns cycle, setting1-3, s1-s21), oldest first."""
        missing = [c for c in ("cycle", *SETTINGS, *SENSORS) if c not in history.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        df = history.sort_values("cycle").reset_index(drop=True).copy()
        df["unit"] = 0
        norm = self.normalizer.transform(df)
        sensors = self.meta["sensors"]
        x, _ = last_windows(norm, sensors, self.meta["window"])

        rul_gbm = float(self.gbm.predict(x)[0])
        rul_lstm = float(self.lstm.predict(x)[0])
        scores = self.monitor.score_frame(norm)
        alarm = self.monitor.alarms(norm)
        first_alarm = norm.loc[alarm, "cycle"]
        latest = norm[sensors].to_numpy()[-1]
        return {
            "subset": self.subset,
            "cycles_observed": int(len(df)),
            "rul_estimate_cycles": round((rul_gbm + rul_lstm) / 2, 1),
            "rul_by_model": {"gradient_boosting": round(rul_gbm, 1), "lstm": round(rul_lstm, 1)},
            "health_score": round(float(scores[-1]), 2),
            "alarm_threshold": round(self.monitor.threshold, 2),
            "alarm": bool(alarm.iloc[-1]),
            "first_alarm_cycle": int(first_alarm.iloc[0]) if not first_alarm.empty else None,
            "score_trend_last_10": round(float(np.polyfit(range(min(10, len(scores))),
                                                          scores[-10:], 1)[0]), 3)
            if len(scores) >= 2 else 0.0,
            "top_contributing_sensors": self.monitor.explain(latest, top=3),
        }


def investigation_report(result: dict) -> str:
    """Plain-language report an engineer can read, built only from the numbers above."""
    lines = [f"Engine assessment ({result['subset']}, {result['cycles_observed']} cycles observed)"]
    lines.append(f"- Estimated remaining useful life: {result['rul_estimate_cycles']} cycles "
                 f"(gradient boosting {result['rul_by_model']['gradient_boosting']}, "
                 f"LSTM {result['rul_by_model']['lstm']})")
    state = "ALARM" if result["alarm"] else "no alarm"
    lines.append(f"- Health score {result['health_score']} vs threshold "
                 f"{result['alarm_threshold']}: {state}"
                 + (f", first raised at cycle {result['first_alarm_cycle']}"
                    if result["first_alarm_cycle"] else ""))
    trend = result["score_trend_last_10"]
    direction = "rising" if trend > 0 else "stable or falling"
    lines.append(f"- Score trend over the last 10 cycles: {direction} ({trend:+.3f} per cycle)")
    lines.append("- Sensors explaining the current deviation:" if result["alarm"]
                 else "- Largest deviations from the healthy state (within normal range):")
    for s in result["top_contributing_sensors"]:
        lines.append(f"    {s['symbol']} ({s['description']}): {s['direction']}, "
                     f"z = {s['z']}, {round(100 * s['share'])}% of the score")
    return "\n".join(lines)
