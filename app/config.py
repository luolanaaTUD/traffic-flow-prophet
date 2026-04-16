from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv_if_present() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_if_present()


class AppConfig(BaseModel):
    default_training_csv: str = Field(default="data/historical_flow_from_summary.csv")
    default_prediction_days: int = Field(default=7, ge=1, le=30)
    qweather_api_key: str = Field(default_factory=lambda: os.getenv("QWEATHER_API_KEY", ""))
    qweather_location: str = Field(default_factory=lambda: os.getenv("QWEATHER_LOCATION", ""))
    qweather_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "QWEATHER_BASE_URL",
            "https://devapi.qweather.com/v7/weather/7d",
        )
    )


config = AppConfig()

