from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ModelMetric(BaseModel):
    model_name: str
    regressors: list[str]
    mae: float | None
    mape: float | None
    holdout_days: int
    status: str


class TrainRequest(BaseModel):
    csv_path: str = Field(default="data/historical_flow.csv")
    holdout_days: int = Field(default=14, ge=7, le=365)
    max_training_days: int = Field(default=1095, ge=30, le=1095)


class TrainResponse(BaseModel):
    model_name: str
    regressors: list[str]
    rows: int
    start_date: date
    end_date: date
    trained_at: datetime
    metrics: list[ModelMetric]


class PredictRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


class PredictionRow(BaseModel):
    ds: date
    yhat: int
    yhat_lower: int
    yhat_upper: int


class PredictResponse(BaseModel):
    model_name: str
    regressors: list[str]
    generated_from: str
    predictions: list[PredictionRow]
