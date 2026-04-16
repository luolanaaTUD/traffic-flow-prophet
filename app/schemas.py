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
    csv_path: str = Field(default="data/historical_flow_from_summary.csv")
    holdout_days: int = Field(default=14, ge=7, le=365)
    max_training_days: int = Field(default=60, ge=60, le=1200)


class TrainResponse(BaseModel):
    model_name: str
    regressors: list[str]
    rows: int
    start_date: date
    end_date: date
    trained_at: datetime
    metrics: list[ModelMetric]
    quality_report: dict | None = None


class PredictRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


class QWeatherDailyForecast(BaseModel):
    fxDate: date
    tempMax: str
    tempMin: str
    windSpeedDay: str
    windSpeedNight: str
    humidity: str
    precip: str
    pressure: str
    vis: str
    cloud: str
    uvIndex: str


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
