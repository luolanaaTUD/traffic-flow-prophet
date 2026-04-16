from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import config
from app.qweather_client import QWeatherClient
from app.schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from app.service import TrafficModelService

app = FastAPI(
    title="Traffic Flow Prophet Backend",
    description="Train traffic models from CSV and predict the next 7 days.",
    version="0.1.0",
)
service = TrafficModelService()
qweather_client = QWeatherClient(
    api_key=config.qweather_api_key,
    location=config.qweather_location,
    base_url=config.qweather_base_url,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "trained": service.is_trained}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    try:
        payload = service.train_from_csv(
            csv_path=req.csv_path,
            holdout_days=req.holdout_days,
            max_training_days=req.max_training_days,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    return TrainResponse(**payload)


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
