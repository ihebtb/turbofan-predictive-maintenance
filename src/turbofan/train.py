"""Train and evaluate every model on C-MAPSS, save artifacts, write reports/results.json.

    python -m turbofan.train                    # FD001 and FD003
    python -m turbofan.train --subsets FD001 FD002 FD003 FD004
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from .anomaly import HealthMonitor, detection_report, inject_faults
from .data import DEFAULT_ROOT, load_subset, split_units
from .metrics import nasa_score, rmse
from .models import GradientBoosting, LinearBaseline, LSTMRegressor
from .preprocess import Normalizer, conditions_for
from .windows import last_windows, training_windows

WINDOW = 30


def fault_injection_report(monitor: HealthMonitor, val, seed: int = 0) -> dict:
    """Per-cycle detection of injected sensor faults on healthy cycles (no smoothing)."""
    healthy = val[val["cycle"] <= monitor.healthy_cycles]
    faulty, labels = inject_faults(healthy, monitor.sensors, rate=0.05, seed=seed)
    point_threshold = float(np.quantile(
        monitor.raw_score(healthy[monitor.sensors].to_numpy()), monitor.quantile))
    flagged = monitor.raw_score(faulty[monitor.sensors].to_numpy()) > point_threshold
    tp = int((flagged & labels).sum())
    return {
        "injected_faults": int(labels.sum()),
        "recall": round(tp / max(labels.sum(), 1), 3),
        "false_positive_rate": round(int((flagged & ~labels).sum()) / max((~labels).sum(), 1), 4),
        "precision": round(tp / max(flagged.sum(), 1), 3),
    }


def run_subset(subset: str, root: Path, out_dir: Path, epochs: int, seed: int = 0) -> dict:
    t0 = time.time()
    train_full, test, test_rul = load_subset(subset, root)
    train, val = split_units(train_full, 0.2, seed)

    norm = Normalizer(n_conditions=conditions_for(subset)).fit(train)
    train_n, val_n, test_n = norm.transform(train), norm.transform(val), norm.transform(test)
    sensors = norm.sensors

    x_tr, y_tr, _ = training_windows(train_n, sensors, WINDOW)
    x_va, y_va, _ = training_windows(val_n, sensors, WINDOW)
    x_te, _ = last_windows(test_n, sensors, WINDOW)

    results, models = {}, {}
    candidates = (LinearBaseline(), GradientBoosting(seed), LSTMRegressor(epochs=epochs, seed=seed))
    for model in candidates:
        model.fit(x_tr, y_tr, x_val=x_va, y_val=y_va)
        pred = model.predict(x_te)
        results[model.name] = {
            "test_rmse": round(rmse(test_rul, pred), 2),
            "test_nasa_score": round(nasa_score(test_rul, pred), 1),
            "val_rmse": round(rmse(y_va, model.predict(x_va)), 2),
        }
        models[model.name] = (model, pred)

    monitor = HealthMonitor().fit(train_n, sensors, calibration=val_n)
    anomaly = {"run_to_failure": detection_report(monitor, val_n),
               "injected_sensor_faults": fault_injection_report(monitor, val_n, seed)}

    target = out_dir / subset
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(norm, target / "normalizer.joblib")
    joblib.dump(models["gradient_boosting"][0], target / "gradient_boosting.joblib")
    torch.save(models["lstm"][0].state(), target / "lstm.pt")
    joblib.dump(monitor, target / "health_monitor.joblib")
    (target / "meta.json").write_text(json.dumps({"subset": subset, "window": WINDOW,
                                                  "sensors": sensors}, indent=2))

    best = min(results, key=lambda k: results[k]["test_rmse"])
    return {
        "subset": subset,
        "train_engines": int(train["unit"].nunique()), "val_engines": int(val["unit"].nunique()),
        "test_engines": int(test["unit"].nunique()), "sensors_used": sensors,
        "rul": results, "best_model": best,
        "lstm_epochs_run": len(models["lstm"][0].history),
        "anomaly": anomaly,
        "seconds": round(time.time() - t0, 1),
        "_plot": {"test_rul": test_rul, "pred": models[best][1], "best": best,
                  "monitor": monitor, "val": val_n},
    }


def plot(summary: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = summary["_plot"]
    order = np.argsort(p["test_rul"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(p["test_rul"][order], "k-", lw=2, label="true RUL")
    ax.plot(p["pred"][order], ".", color="tab:blue", label=f"predicted ({p['best']})")
    ax.set(xlabel="test engines, sorted by true RUL", ylabel="remaining cycles",
           title=f"{summary['subset']}: remaining useful life on the official test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{summary['subset']}_rul.png", dpi=130)
    plt.close(fig)

    monitor, val = p["monitor"], p["val"]
    scores = monitor.score_frame(val)
    fig, ax = plt.subplots(figsize=(9, 4))
    for unit in val["unit"].unique()[:6]:
        mask = (val["unit"] == unit).to_numpy()
        cycles = val.loc[mask, "cycle"].to_numpy()
        ax.plot(cycles - cycles.max(), scores[mask], lw=1, label=f"engine {unit}")
    ax.axhline(monitor.threshold, color="red", ls="--", label="alarm threshold")
    ax.set_yscale("log")
    ax.set(xlabel="cycles before failure", ylabel="anomaly score (Mahalanobis², log)",
           title=f"{summary['subset']}: health score of validation engines")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{summary['subset']}_health.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--models", default="models")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    out = {}
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        summary = run_subset(subset, Path(args.root), Path(args.models), args.epochs)
        plot(summary, fig_dir)
        summary.pop("_plot")
        out[subset] = summary
        shown = {k: summary[k] for k in ("subset", "rul", "anomaly", "seconds")}
        print(json.dumps(shown, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
