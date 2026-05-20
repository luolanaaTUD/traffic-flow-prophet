from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

# Prophet imports plotly for optional notebooks; suppress the error log on API startup.
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

from fastapi import FastAPI, HTTPException

from app.config import config
from app.qweather_client import QWeatherClient
from app.schemas import (
    PredictRequest,
    PredictResponse,
    TrainFromCsvRequest,
    TrainRequest,
    TrainResponse,
)
from app.service import TrafficModelService

_startup_log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _startup_log.info("Swagger UI:  http://127.0.0.1:8000/docs")
    _startup_log.info("ReDoc:       http://127.0.0.1:8000/redoc")
    _startup_log.info("OpenAPI:     http://127.0.0.1:8000/openapi.json")
    yield


app = FastAPI(
    title="Traffic Flow Prophet Backend",
    description="Train traffic models from JSON records or CSV and predict the next 7 days.",
    version="0.1.0",
    lifespan=lifespan,
)
service = TrafficModelService()
qweather_client = QWeatherClient(
    api_key=config.qweather_api_key,
    location=config.qweather_location,
    base_url=config.qweather_base_url,
)


def _run_train(train_fn: Callable[[], dict]) -> TrainResponse:
    try:
        payload = train_fn()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    return TrainResponse(**payload)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "trained": service.is_trained}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    return _run_train(
        lambda: service.train_from_records(
            records=[record.model_dump(mode="json") for record in req.records],
            holdout_days=req.holdout_days,
            max_training_days=req.max_training_days,
        )
    )


@app.post("/train/from-csv", response_model=TrainResponse)
def train_from_csv(req: TrainFromCsvRequest) -> TrainResponse:
    return _run_train(
        lambda: service.train_from_csv(
            csv_path=req.csv_path,
            holdout_days=req.holdout_days,
            max_training_days=req.max_training_days,
        )
    )


@app.post("/predict/next-7-days", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    days = req.days or config.default_prediction_days
    try:
        qweather_rows = qweather_client.fetch_daily_forecast()
        payload = service.predict_next_days(forecast_rows=qweather_rows, days=days)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(**payload)
