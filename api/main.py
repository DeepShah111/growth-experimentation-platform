"""REST API of the Growth Experimentation Platform.

    uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs

POST /analyze        JSON records + column mapping  -> SRM, lift, CUPED, Holm, decision per treatment arm
POST /analyze/csv    CSV upload (multipart) + the same mapping as form fields
POST /plan           sample size, duration, MDE and power curve for a planned test
POST /peeking        false-positive rate of repeated looks at an A/A test
GET  /datasets       the bundled datasets and their columns
GET  /health
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from src.engine import planner, sequential
from src.engine.analyze import AnalysisConfig, AnalysisError, analyze_experiment, to_jsonable
from src.engine.datasets import (CRITEO, HILLSTROM, MAX_UPLOAD_ROWS, criteo_available, hillstrom_available,
                                 read_upload)

app = FastAPI(title="Growth Experimentation Platform API", version="2.0.0",
              description="Turn any randomised experiment table into a trust-checked ship / iterate / hold decision.")

MAX_RECORDS = 200_000
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


class ConfigIn(BaseModel):
    control: str = "control"
    alpha: float = Field(0.05, gt=0, lt=0.5)
    srm_alpha: float = Field(0.001, gt=0, lt=0.5)
    min_effect: float = Field(0.03, ge=0, lt=5)
    power: float = Field(0.8, gt=0.5, lt=0.999)
    planned_split: dict[str, float] | None = None
    value_per_unit: float | None = None
    cost_per_treated: float = Field(0.0, ge=0)
    currency: str = Field("USD", max_length=5)
    guardrail: str | None = None

    def to_cfg(self) -> AnalysisConfig:
        return AnalysisConfig(**self.model_dump())


class AnalyzeIn(BaseModel):
    records: list[dict] = Field(..., description="one object per user, e.g. {'variant': 'treatment', 'orders': 2, 'orders_pre': 1}")
    variant: str
    outcome: str
    covariates: list[str] = []
    config: ConfigIn = ConfigIn()

    @field_validator("records")
    @classmethod
    def _size(cls, v):
        if len(v) > MAX_RECORDS:
            raise ValueError(f"at most {MAX_RECORDS:,} records per request; use /analyze/csv or sample your data")
        return v


class PlanIn(BaseModel):
    kind: str = Field("rate", pattern="^(rate|mean)$")
    baseline: float = Field(..., gt=0)
    rel_lift: float = Field(..., gt=0, lt=10)
    sd: float | None = Field(None, gt=0)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    power: float = Field(0.8, gt=0.5, lt=0.999)
    rho: float = Field(0.0, ge=0, lt=1)
    daily_users: float | None = Field(None, gt=0)
    n_arms: int = Field(2, ge=2, le=10)


class PeekIn(BaseModel):
    n_looks: int = Field(..., ge=1, le=60)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    n_sims: int = Field(20_000, ge=1_000, le=100_000)


def _run(df: pd.DataFrame, variant: str, outcome: str, covariates: list[str], cfg: AnalysisConfig) -> dict:
    try:
        return to_jsonable(analyze_experiment(df, variant, outcome, covariates, cfg))
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/datasets")
def datasets() -> list[dict]:
    out = []
    for spec, ok in ((HILLSTROM, hillstrom_available()), (CRITEO, criteo_available())):
        out.append({"key": spec.key, "label": spec.label, "available": ok, "variant": spec.variant, "control": spec.control,
                    "outcomes": spec.outcomes, "covariates": list(spec.covariates), "description": spec.description,
                    "caveats": spec.caveats})
    return out


@app.post("/analyze")
def analyze(body: AnalyzeIn) -> dict:
    df = pd.DataFrame.from_records(body.records)
    return _run(df, body.variant, body.outcome, body.covariates, body.config.to_cfg())


@app.post("/analyze/csv")
async def analyze_csv(file: UploadFile = File(...), variant: str = Form(...), outcome: str = Form(...),
                      covariates: str = Form(""), config: str = Form("{}")) -> dict:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        df = read_upload(raw, MAX_UPLOAD_ROWS)
        cfg = ConfigIn(**json.loads(config or "{}")).to_cfg()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    covs = [c.strip() for c in covariates.split(",") if c.strip()]
    return _run(df, variant, outcome, covs, cfg)


@app.post("/plan")
def plan(body: PlanIn) -> dict:
    try:
        p = planner.plan(body.kind, body.baseline, body.rel_lift, body.sd, body.alpha, body.power, body.rho,
                         body.daily_users, body.n_arms)
        lifts = np.linspace(body.rel_lift / 4, body.rel_lift * 2, 12)
        curve = planner.power_curve(body.kind, body.baseline, p["n_per_arm"], lifts, body.sd, body.alpha, body.rho)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**p, "power_curve": curve.round(4).to_dict("records")}


@app.post("/peeking")
def peeking(body: PeekIn) -> dict:
    return {"results": sequential.peeking_study(body.n_looks, body.alpha, body.n_sims).round(4).to_dict("records")}
