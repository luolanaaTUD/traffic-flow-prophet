from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

import pandas as pd
from prophet import Prophet

from app.data_processing import (
    build_future_features_from_history,
    enrich_weather_features,
    load_future_features_csv,
    load_training_csv,
    validate_training_df,
)
from app.modeling import EvaluationResult, train_multi_model


@dataclass
class ModelArtifacts:
    model: Prophet
    model_name: str
    regressors: list[str]
    train_df: pd.DataFrame
    metrics: list[EvaluationResult]
    trained_at: datetime


class TrafficModelService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._artifacts: ModelArtifacts | None = None

    @property
    def is_trained(self) -> bool:
        return self._artifacts is not None

    def train_from_csv(
        self,
        csv_path: str,
        holdout_days: int = 14,
        max_training_days: int = 1095,
    ) -> dict:
        raw_df = load_training_csv(csv_path)
        train_df = validate_training_df(raw_df)
        train_df = enrich_weather_features(train_df)
        if max_training_days > 0:
            latest_date = train_df["ds"].max().normalize()
            cutoff_date = latest_date - pd.Timedelta(days=max_training_days - 1)
            train_df = train_df[train_df["ds"] >= cutoff_date].copy()
            if train_df.empty:
                raise ValueError("No rows left after applying max_training_days window.")

        model, model_name, regressors, metrics = train_multi_model(
            df=train_df,
            holdout_days=holdout_days,
        )

        artifacts = ModelArtifacts(
            model=model,
            model_name=model_name,
            regressors=regressors,
            train_df=train_df,
            metrics=metrics,
            trained_at=datetime.now(UTC),
        )

        with self._lock:
            self._artifacts = artifacts

        return {
            "model_name": model_name,
            "regressors": regressors,
            "rows": int(len(train_df)),
            "start_date": train_df["ds"].min().date(),
            "end_date": train_df["ds"].max().date(),
            "trained_at": artifacts.trained_at,
            "metrics": [
                {
                    "model_name": item.model_name,
                    "regressors": item.regressors,
                    "mae": item.mae,
                    "mape": item.mape,
                    "holdout_days": item.holdout_days,
                    "status": item.status,
                }
                for item in metrics
            ],
        }

    def predict_next_days(self, days: int = 7, future_csv_path: str | None = None) -> dict:
        with self._lock:
            artifacts = self._artifacts

        if artifacts is None:
            raise RuntimeError("Model is not trained. Call /train first.")

        if future_csv_path:
            future_df = load_future_features_csv(future_csv_path)
            generated_from = "future_csv"
        else:
            future_df = build_future_features_from_history(
                history_df=artifacts.train_df,
                regressors=artifacts.regressors,
                days=days,
            )
            generated_from = "historical_profile"

        if days and len(future_df) < days:
            raise ValueError(f"Future features must contain at least {days} rows.")

        missing_cols = [col for col in artifacts.regressors if col not in future_df.columns]
        if missing_cols:
            raise ValueError(f"Future features missing required columns: {missing_cols}")

        future_for_model = future_df[["ds"] + artifacts.regressors].copy()
        if days:
            future_for_model = future_for_model.head(days).copy()

        forecast = artifacts.model.predict(future_for_model)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            result[col] = result[col].clip(lower=0).round().astype(int)

        predictions = [
            {
                "ds": row.ds.date(),
                "yhat": int(row.yhat),
                "yhat_lower": int(row.yhat_lower),
                "yhat_upper": int(row.yhat_upper),
            }
            for row in result.itertuples(index=False)
        ]

        return {
            "model_name": artifacts.model_name,
            "regressors": artifacts.regressors,
            "generated_from": generated_from,
            "predictions": predictions,
        }
