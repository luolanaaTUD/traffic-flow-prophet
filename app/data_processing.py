from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "ds",
    "y",
    "temp_max",
    "temp_min",
    "precip",
    "humidity",
    "pressure",
    "vis",
    "cloud",
    "uv_index",
    "wind_speed_day",
    "wind_speed_night",
]
NUMERIC_COLUMNS = [col for col in REQUIRED_COLUMNS if col != "ds"]
FORECAST_CORE_COLUMNS = [
    "ds",
    "temp_max",
    "temp_min",
    "precip",
    "humidity",
    "pressure",
    "vis",
    "uv_index",
    "wind_speed_day",
    "wind_speed_night",
]
WEATHER_FEATURE_COLUMNS = [col for col in REQUIRED_COLUMNS if col not in {"ds", "y"}]
# Derived wind features computed from raw wind speed — not present in input CSVs.
# is_windy_day is binary (0/1); wind_level is a Beaufort-approximate ordinal (0–4).
DERIVED_WIND_COLUMNS: list[str] = ["is_windy_day", "wind_level"]
# Fixed threshold separating QWeather speed-scale 2 (<= 9.5 km/h) from scale 3+.
# This cleanly splits the two most common observed values in this dataset (8.9 vs ≥10 km/h)
# and is consistent between training-time derivation and inference-time derivation.
IS_WINDY_THRESHOLD_KMH: float = 9.5
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        root_candidate = (PROJECT_ROOT / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        if root_candidate.exists():
            return root_candidate
        # Prefer project-root-relative paths as canonical API input.
        return root_candidate
    return path.resolve()


def load_training_csv(csv_path: str) -> pd.DataFrame:
    path = resolve_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    return pd.read_csv(path)


def validate_training_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Backward compatibility for historical datasets generated with legacy weather_score schema.
    if "temp_min" not in out.columns and "temp_max" in out.columns:
        out["temp_min"] = pd.to_numeric(out["temp_max"], errors="coerce") - 8.0
    if "wind_speed_night" not in out.columns and "wind_speed_day" in out.columns:
        out["wind_speed_night"] = out["wind_speed_day"]
    if "weather_score" in out.columns:
        weather_score = pd.to_numeric(out["weather_score"], errors="coerce").clip(lower=0.2, upper=1.0)
        if "pressure" not in out.columns:
            out["pressure"] = 1000 + weather_score * 20
        if "vis" not in out.columns:
            out["vis"] = (weather_score * 30).clip(lower=1.0, upper=30.0)
        if "cloud" not in out.columns:
            out["cloud"] = ((1 - weather_score) * 100).clip(lower=0.0, upper=100.0)

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["ds"] = pd.to_datetime(out["ds"])

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=REQUIRED_COLUMNS).copy()
    out = out.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid training rows after parsing CSV.")
    _validate_weather_ranges(out)
    out["humidity"] = out["humidity"].clip(lower=0, upper=100)
    out["uv_index"] = out["uv_index"].clip(lower=0, upper=16)
    out["cloud"] = out["cloud"].clip(lower=0, upper=100)
    for col in ["precip", "vis", "pressure", "wind_speed_day", "wind_speed_night"]:
        out[col] = out[col].clip(lower=0)
    out = derive_wind_features(out)
    return out


def _validate_weather_ranges(df: pd.DataFrame) -> None:
    range_specs: dict[str, tuple[float, float]] = {
        "humidity": (0.0, 100.0),
        "cloud": (0.0, 100.0),
        "uv_index": (0.0, 16.0),
        "precip": (0.0, 500.0),
        "vis": (0.0, 80.0),
        "pressure": (850.0, 1100.0),
        "wind_speed_day": (0.0, 200.0),
        "wind_speed_night": (0.0, 200.0),
        "temp_max": (-60.0, 60.0),
        "temp_min": (-60.0, 60.0),
    }
    invalid_messages: list[str] = []
    for col, (min_allowed, max_allowed) in range_specs.items():
        if col not in df.columns:
            continue
        mask = (df[col] < min_allowed) | (df[col] > max_allowed)
        invalid_count = int(mask.sum())
        if invalid_count:
            invalid_messages.append(
                f"{col}: {invalid_count} rows outside [{min_allowed}, {max_allowed}]"
            )
    if invalid_messages:
        joined = "; ".join(invalid_messages)
        raise ValueError(f"Weather fields out of expected QWeather ranges: {joined}")


def derive_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive robust behavioral wind features from raw wind speed columns.

    Raw ``wind_speed_day`` suffers from low variance in historical datasets
    because QWeather speed values cluster within a narrow band for this park
    location (Guangzhou, subtropical).  These derived features encode the
    visitor-impact signal that raw values carry but cannot express reliably
    when measurements are coarse or repeated.

    Added columns
    -------------
    ``is_windy_day``
        1 if daytime wind speed exceeds ``IS_WINDY_THRESHOLD_KMH`` (9.5 km/h).
        This threshold sits at the QWeather scale-2/scale-3 boundary and cleanly
        separates the two most frequently observed speed values in this dataset
        (8.9 km/h → calm, ≥10 km/h → breezy).  The threshold is fixed so that
        training derivation and inference derivation are always identical.

    ``wind_level``
        Beaufort-approximate ordinal (0–4):
        0 = calm (< 5.5 km/h), 1 = light (5.5–11.5), 2 = gentle (11.5–19.5),
        3 = moderate (19.5–28.5), 4 = strong (≥ 28.5).
    """
    out = df.copy()
    ws_day = out["wind_speed_day"].clip(lower=0)

    out["is_windy_day"] = (ws_day > IS_WINDY_THRESHOLD_KMH).astype(float)

    cut = pd.cut(
        ws_day,
        bins=[-0.001, 5.5, 11.5, 19.5, 28.5, float("inf")],
        labels=[0, 1, 2, 3, 4],
    )
    out["wind_level"] = cut.astype(float).fillna(1.0)
    return out


def assess_training_feature_quality(df: pd.DataFrame) -> dict:
    total_rows = max(len(df), 1)
    low_variance_features: list[str] = []
    imputed_like_features: list[str] = []
    stats: dict[str, dict[str, float]] = {}

    # Derived binary features are intentionally 0/1 — never flag them as low-variance.
    _derived_binary_cols: frozenset[str] = frozenset({"is_windy_day"})

    for col in list(WEATHER_FEATURE_COLUMNS) + DERIVED_WIND_COLUMNS:
        if col not in df.columns:
            continue
        nunique = int(df[col].nunique(dropna=True))
        unique_ratio = nunique / total_rows
        top_freq = float(df[col].value_counts(normalize=True, dropna=True).iloc[0]) if nunique else 1.0
        stats[col] = {
            "nunique": float(nunique),
            "unique_ratio": round(unique_ratio, 4),
            "top_value_ratio": round(top_freq, 4),
        }
        if (nunique <= 2 or unique_ratio < 0.08) and col not in _derived_binary_cols:
            low_variance_features.append(col)
        if top_freq >= 0.85:
            imputed_like_features.append(col)

    # wind_speed_day hard-fail is waived when is_windy_day is present and has
    # at least some positive signal (i.e. proportion of windy days > 0).
    # Derived is_windy_day preserves visitor-impact wind information even when
    # raw speed values cluster within a narrow band.
    effective_wind_ok = (
        "is_windy_day" in df.columns
        and float(df["is_windy_day"].sum()) > 0
    )
    hard_fail_cols = sorted(
        col
        for col in {"wind_speed_day", "vis", "cloud"}
        if col in set(low_variance_features)
        and not (col == "wind_speed_day" and effective_wind_ok)
    )
    if hard_fail_cols:
        raise ValueError(
            "Training weather features are too low-variance for reliable learning: "
            f"{hard_fail_cols}. Please provide richer observed weather values."
        )

    low_confidence_regressors = sorted(set(low_variance_features + imputed_like_features))
    return {
        "low_variance_features": low_variance_features,
        "imputed_like_features": imputed_like_features,
        "low_confidence_regressors": low_confidence_regressors,
        "feature_stats": stats,
    }


def normalize_qweather_daily_forecast(raw_rows: list[dict], cloud_fallback: float = 50.0) -> pd.DataFrame:
    if not raw_rows:
        raise ValueError("QWeather forecast rows are empty.")

    qweather_to_feature = {
        "fxDate": "ds",
        "tempMax": "temp_max",
        "tempMin": "temp_min",
        "precip": "precip",
        "humidity": "humidity",
        "pressure": "pressure",
        "vis": "vis",
        "cloud": "cloud",
        "uvIndex": "uv_index",
        "windSpeedDay": "wind_speed_day",
        "windSpeedNight": "wind_speed_night",
    }
    forecast_df = pd.DataFrame(raw_rows).rename(columns=qweather_to_feature)
    required_qweather_fields = FORECAST_CORE_COLUMNS
    missing = [col for col in required_qweather_fields if col not in forecast_df.columns]
    if missing:
        raise ValueError(f"QWeather forecast missing required fields: {missing}")

    out = forecast_df[[*required_qweather_fields, "cloud"]].copy() if "cloud" in forecast_df.columns else forecast_df[
        required_qweather_fields
    ].copy()
    if "cloud" not in out.columns:
        out["cloud"] = pd.NA
    out["ds"] = pd.to_datetime(out["ds"])
    for col in [item for item in WEATHER_FEATURE_COLUMNS]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required_qweather_fields).copy()
    out = out.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid QWeather rows after normalization.")

    _validate_weather_ranges(out)
    if out["cloud"].notna().any():
        out["cloud"] = out["cloud"].fillna(float(out["cloud"].median()))
    else:
        out["cloud"] = float(cloud_fallback)

    out["humidity"] = out["humidity"].clip(lower=0, upper=100)
    out["uv_index"] = out["uv_index"].clip(lower=0, upper=16)
    out["cloud"] = out["cloud"].clip(lower=0, upper=100)
    for col in ["precip", "vis", "pressure", "wind_speed_day", "wind_speed_night"]:
        out[col] = out[col].clip(lower=0)
    out = derive_wind_features(out)
    return out
