import json

import joblib
import torch
from fastapi.testclient import TestClient

from turbofan import api
from turbofan.anomaly import HealthMonitor
from turbofan.data import SENSORS, SETTINGS, add_rul
from turbofan.models import GradientBoosting, LSTMRegressor
from turbofan.preprocess import Normalizer
from turbofan.service import EngineAssessor, investigation_report
from turbofan.windows import training_windows

from .conftest import make_fleet


def _train_artifacts(model_dir):
    fleet = add_rul(make_fleet(16, seed=3))
    norm = Normalizer().fit(fleet)
    data = norm.transform(fleet)
    x, y, _ = training_windows(data, norm.sensors, 30)
    target = model_dir / "FD001"
    target.mkdir(parents=True)
    joblib.dump(norm, target / "normalizer.joblib")
    joblib.dump(GradientBoosting().fit(x, y), target / "gradient_boosting.joblib")
    torch.save(LSTMRegressor(hidden=8, epochs=1).fit(x, y).state(), target / "lstm.pt")
    joblib.dump(HealthMonitor().fit(data, norm.sensors), target / "health_monitor.joblib")
    (target / "meta.json").write_text(json.dumps({"window": 30, "sensors": norm.sensors}))
    return fleet


def test_assessment_of_a_failing_engine(tmp_path):
    fleet = _train_artifacts(tmp_path)
    engine = fleet[fleet["unit"] == 1]
    result = EngineAssessor(tmp_path, "FD001").assess(engine)
    assert result["alarm"] is True
    assert result["rul_estimate_cycles"] < 40
    assert len(result["top_contributing_sensors"]) == 3
    assert "Estimated remaining useful life" in investigation_report(result)


def test_api_assess(tmp_path, monkeypatch):
    fleet = _train_artifacts(tmp_path)
    monkeypatch.setattr(api, "MODEL_DIR", str(tmp_path))
    api.assessor.cache_clear()
    engine = fleet[fleet["unit"] == 2][["cycle", *SETTINGS, *SENSORS]]
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/assess", json={"subset": "FD001",
                                         "cycles": engine.to_dict(orient="records")})
        assert r.status_code == 200
        body = r.json()
        assert {"rul_estimate_cycles", "alarm", "report"} <= body.keys()
        bad = client.post("/assess", json={"subset": "FD009", "cycles": []})
        assert bad.status_code == 422
