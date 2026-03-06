from __future__ import annotations

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    default_training_csv: str = Field(default="data/historical_flow.csv")
    default_prediction_days: int = Field(default=7, ge=1, le=30)


config = AppConfig()

