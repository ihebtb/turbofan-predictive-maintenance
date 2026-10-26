import numpy as np

from turbofan.anomaly import HealthMonitor, detection_report, inject_faults
from turbofan.data import split_units
from turbofan.metrics import rmse
from turbofan.models import GradientBoosting, LinearBaseline, LSTMRegressor
from turbofan.preprocess import Normalizer
from turbofan.windows import training_windows


def _prepared(fleet):
    train, val = split_units(fleet, 0.25, seed=0)
    norm = Normalizer().fit(train)
    return norm, norm.transform(train), norm.transform(val)


def test_models_learn_remaining_life(fleet):
    norm, train, val = _prepared(fleet)
    x, y, _ = training_windows(train, norm.sensors, 30)
    xv, yv, _ = training_windows(val, norm.sensors, 30)
    naive = rmse(yv, np.full_like(yv, y.mean()))
    for model in (LinearBaseline(), GradientBoosting()):
        model.fit(x, y)
        assert rmse(yv, model.predict(xv)) < 0.6 * naive
    lstm = LSTMRegressor(hidden=16, epochs=3).fit(x, y, x_val=xv, y_val=yv)
    assert lstm.predict(xv).shape == yv.shape
    assert lstm.history and "val_rmse" in lstm.history[0]
    clone = LSTMRegressor.from_state(lstm.state())
    assert np.allclose(clone.predict(xv[:5]), lstm.predict(xv[:5]))


def test_health_monitor_alarms_before_failure_not_when_healthy(fleet):
    norm, train, val = _prepared(fleet)
    monitor = HealthMonitor().fit(train, norm.sensors, calibration=val)
    report = detection_report(monitor, val)
    assert report["detected_before_failure"] == 1.0
    assert report["alarm_in_healthy_phase"] == 0.0
    assert report["median_lead_time_cycles"] > 5


def test_explanations_point_at_degrading_sensors(fleet):
    norm, train, val = _prepared(fleet)
    monitor = HealthMonitor().fit(train, norm.sensors, calibration=val)
    last_cycle = val.groupby("unit").tail(1)[norm.sensors].to_numpy()[0]
    top = monitor.explain(last_cycle, top=3)
    assert all(t["sensor"] in {"s2", "s3", "s4", "s7", "s9", "s11", "s12", "s14"} for t in top)
    shares = monitor.contributions(last_cycle).iloc[0]
    assert np.isclose(shares.sum(), 1.0)


def test_injected_faults_are_labelled(fleet):
    norm, train, _ = _prepared(fleet)
    healthy = train[train["cycle"] <= 30]
    faulty, labels = inject_faults(healthy, norm.sensors, rate=0.1, seed=0)
    assert labels.sum() == int(len(healthy) * 0.1)
    changed = (faulty[norm.sensors] != healthy[norm.sensors]).any(axis=1).to_numpy()
    assert (changed == labels).all()
