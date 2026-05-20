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
# is_windy_day is binary (0/1); wind_level is the standard Beaufort scale ordinal (0–12).
DERIVED_WIND_COLUMNS: list[str] = ["is_windy_day", "wind_level"]
# Beaufort scale 3 (微风 / Gentle breeze) onset: 12 km/h.
# At this speed flags visibly flutter (旌旗展开) — the first level where park visitors
# consistently notice the wind.  The threshold is fixed so that training derivation
# and inference derivation are always identical.
IS_WINDY_THRESHOLD_KMH: float = 12.0
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


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        raise ValueError("Training records must not be empty.")
    return pd.DataFrame(records)


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
        1 if daytime wind speed reaches Beaufort scale 3 (微风 / Gentle breeze),
        i.e. ``wind_speed_day >= IS_WINDY_THRESHOLD_KMH`` (12 km/h).  At this
        level flags visibly flutter (旌旗展开) — the first Beaufort grade where
        park visitors consistently feel the wind.  When all observed values fall
        below this threshold (persistently calm conditions), ``is_windy_day``
        will be all-zero and will be treated as a low-confidence regressor.

    ``wind_level``
        Standard Beaufort scale integer (0–12), derived from ``wind_speed_day``
        using the official km/h boundaries::

            0  无风  Calm            < 2 km/h
            1  软风  Light air       2–5 km/h
            2  轻风  Light breeze    6–11 km/h
            3  微风  Gentle breeze   12–19 km/h
            4  和风  Moderate        20–28 km/h
            5  清风  Fresh breeze    29–38 km/h
            6  强风  Strong breeze   39–49 km/h
            7  疾风  Near gale       50–61 km/h
            8  大风  Gale            62–74 km/h
            9  烈风  Strong gale     75–88 km/h
           10  狂风  Storm           89–102 km/h
           11  暴风  Violent storm   103–117 km/h
           12  飓风  Hurricane       ≥ 118 km/h
    """
    out = df.copy()
    ws_day = out["wind_speed_day"].clip(lower=0)

    out["is_windy_day"] = (ws_day >= IS_WINDY_THRESHOLD_KMH).astype(float)

    # Official Beaufort scale km/h breakpoints (Beaufort 0 through 12).
    # right=False → [left, right) intervals so each grade's lower bound is
    # inclusive (e.g. 12 km/h → Beaufort 3, not Beaufort 2).
    cut = pd.cut(
        ws_day,
        bins=[-0.001, 2, 6, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118, float("inf")],
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        right=False,
    )
    out["wind_level"] = cut.astype(float).fillna(1.0)
    return out


def assess_training_feature_quality(df: pd.DataFrame) -> dict:
    total_rows = max(len(df), 1)
    low_variance_features: list[str] = []
    imputed_like_features: list[str] = []
    stats: dict[str, dict[str, float]] = {}

    # is_windy_day is a designed binary (0/1) feature.  Only suppress the
    # low-variance flag when BOTH values are actually present (proper binary split).
    # If it is constant (e.g. all-zero because all wind < 12 km/h), treat it
    # normally as low-variance so its prior_scale is reduced to 0.05.
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
        is_proper_binary = col in _derived_binary_cols and nunique == 2
        if (nunique <= 2 or unique_ratio < 0.08) and not is_proper_binary:
            low_variance_features.append(col)
        if top_freq >= 0.85:
            imputed_like_features.append(col)

    # wind_speed_day hard-fail is waived whenever Beaufort-level wind features
    # have been successfully derived (wind_level column is present).
    # Flat or calm wind data is valid training data — it simply means the park
    # experienced consistently low-wind conditions.  The low-confidence path
    # automatically down-weights flat wind regressors via reduced prior_scale.
    effective_wind_ok = "wind_level" in df.columns
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
