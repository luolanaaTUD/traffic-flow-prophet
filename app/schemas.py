from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelMetric(BaseModel):
    model_name: str
    regressors: list[str]
    mae: float | None
    mape: float | None
    holdout_days: int
    status: str


class TrainingRecord(BaseModel):
    ds: date
    y: float
    temp_max: float
    temp_min: float
    precip: float
    humidity: float
    pressure: float
    vis: float
    cloud: float
    uv_index: float
    wind_speed_day: float
    wind_speed_night: float


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[TrainingRecord] = Field(min_length=1)
    holdout_days: int = Field(default=14, ge=7, le=365)
    max_training_days: int = Field(default=120, ge=60, le=720)


class TrainFromCsvRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: str = Field(default="data/historical_flow_from_summary.csv")
    holdout_days: int = Field(default=14, ge=7, le=365)
    max_training_days: int = Field(default=120, ge=60, le=720)


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
