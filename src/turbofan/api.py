"""FastAPI service: send an engine's sensor history, get remaining life, health score,
alarm state and an investigation report.

    uvicorn turbofan.api:app --reload      then open http://localhost:8000/docs
"""

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import SENSORS, SETTINGS
from .service import EngineAssessor, investigation_report

MODEL_DIR = os.environ.get("TURBOFAN_MODELS", "models")

app = FastAPI(title="Turbofan predictive maintenance", version="0.1.0")


class Cycle(BaseModel):
    cycle: int
    setting1: float
    setting2: float
    setting3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float


class AssessRequest(BaseModel):
    subset: str = Field("FD001", pattern="^FD00[1-4]$")
    cycles: list[Cycle] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=4)
def assessor(subset: str) -> EngineAssessor:
    return EngineAssessor(MODEL_DIR, subset)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/assess")
def assess(body: AssessRequest) -> dict:
    try:
        engine = assessor(body.subset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    history = pd.DataFrame([c.model_dump() for c in body.cycles])[["cycle", *SETTINGS, *SENSORS]]
    result = engine.assess(history)
    return {**result, "report": investigation_report(result)}
