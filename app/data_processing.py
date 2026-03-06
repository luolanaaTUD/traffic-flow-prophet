from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["ds", "y", "temp_max", "weather_score"]
OPTIONAL_FEATURE_COLUMNS = [
    "precip",
    "wind_speed_day",
    "humidity",
    "uv_index",
    "is_rain",
    "is_severe_weather",
]
NUMERIC_COLUMNS = ["y", "temp_max", "weather_score"] + OPTIONAL_FEATURE_COLUMNS


def resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_training_csv(csv_path: str) -> pd.DataFrame:
    path = resolve_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    return pd.read_csv(path)


def validate_training_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["ds", "y", "temp_max", "weather_score"]).copy()
    out = out.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid training rows after parsing CSV.")
    return out


def enrich_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "precip" not in out.columns:
        out["precip"] = np.select(
            [
                out["weather_score"] <= 0.30,
                out["weather_score"] <= 0.60,
                out["weather_score"] <= 0.85,
            ],
            [15.0, 5.0, 1.0],
            default=0.0,
        )
    out["precip"] = out["precip"].fillna(0.0).clip(lower=0.0)

    if "wind_speed_day" not in out.columns:
        out["wind_speed_day"] = 3.0 + (1 - out["weather_score"]).clip(lower=0) * 6
    out["wind_speed_day"] = out["wind_speed_day"].fillna(out["wind_speed_day"].median()).clip(lower=0.0)

    if "humidity" not in out.columns:
        out["humidity"] = 60 + (1 - out["weather_score"]).clip(lower=0) * 25
    out["humidity"] = out["humidity"].fillna(65).clip(lower=10, upper=100)

    if "uv_index" not in out.columns:
        out["uv_index"] = (out["temp_max"] - 10) / 2
    out["uv_index"] = out["uv_index"].fillna(out["uv_index"].median()).clip(lower=0, upper=12)

    out["is_rain"] = ((out["precip"] > 0.0) | (out["weather_score"] < 0.75)).astype(int)
    out["is_severe_weather"] = ((out["precip"] >= 10.0) | (out["weather_score"] <= 0.35)).astype(int)
    return out


def load_future_features_csv(csv_path: str) -> pd.DataFrame:
    path = resolve_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Future-features CSV not found: {path}")

    out = pd.read_csv(path).copy()
    if "ds" not in out.columns:
        raise ValueError("Future-features CSV must include `ds` column.")

    out["ds"] = pd.to_datetime(out["ds"])
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # If core weather features are present, generate optional columns as needed.
    if "temp_max" in out.columns and "weather_score" in out.columns:
        out = enrich_weather_features(out)

    out = out.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    return out


def build_future_features_from_history(
    history_df: pd.DataFrame,
    regressors: list[str],
    days: int,
) -> pd.DataFrame:
    if days < 1:
        raise ValueError("`days` must be >= 1")

    history = history_df.copy()
    history["dow"] = history["ds"].dt.dayofweek

    start_date = history["ds"].max().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=days, freq="D")
    future = pd.DataFrame({"ds": future_dates})
    future["dow"] = future["ds"].dt.dayofweek

    for col in regressors:
        if col not in history.columns:
            raise ValueError(f"Historical data missing required regressor: {col}")
        dow_avg = history.groupby("dow")[col].mean()
        overall_avg = float(history[col].mean())
        future[col] = future["dow"].map(dow_avg).fillna(overall_avg)

    if "is_rain" in future.columns:
        future["is_rain"] = future["is_rain"].round().clip(lower=0, upper=1).astype(int)
    if "is_severe_weather" in future.columns:
        future["is_severe_weather"] = (
            future["is_severe_weather"].round().clip(lower=0, upper=1).astype(int)
        )

    if "precip" in future.columns:
        future["precip"] = future["precip"].clip(lower=0.0)
    if "wind_speed_day" in future.columns:
        future["wind_speed_day"] = future["wind_speed_day"].clip(lower=0.0)
    if "humidity" in future.columns:
        future["humidity"] = future["humidity"].clip(lower=10.0, upper=100.0)
    if "uv_index" in future.columns:
        future["uv_index"] = future["uv_index"].clip(lower=0.0, upper=12.0)
    if "weather_score" in future.columns:
        future["weather_score"] = future["weather_score"].clip(lower=0.2, upper=1.0)

    return future.drop(columns=["dow"])

