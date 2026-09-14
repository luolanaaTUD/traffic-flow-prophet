# Traffic Flow Prophet Backend

Chinese version: [README.zh-CN.md](README.zh-CN.md)

This backend is extracted from the notebook workflow and provides:

- model training from JSON records (production) or historical traffic CSV (local dev)
- next-7-days traffic prediction API
- **NEW**: Optional TimesFM-3 forecasting backend alongside Prophet

The default training model is `multi_weather_regressors` (Prophet). You can optionally select `timesfm_weather_holiday` (TimesFM-3) via the `model_name` parameter.

## Model Selection

### Prophet (Production-Safe Default)
- Model name: `multi_weather_regressors`
- License: MIT (production-safe)
- Baseline model for traffic forecasting

### TimesFM-3 (Experimental, Non-Commercial)
- Model name: `timesfm_weather_holiday`
- Checkpoint: `google/timesfm-3.0-pytorch`
- License: **TimesFM Non-Commercial License** (non-commercial / non-production use only)
- Requires: `pip install -e ".[timesfm]"` or `pip install timesfm[torch] torch>=2.0.0`
- Features:
  - Weather covariates (same as Prophet: temp_max, temp_min, precip, humidity, pressure, vis, cloud, uv_index, wind_speed_day, wind_speed_night, is_windy_day)
  - CN holidays from Prophet's calendar (`make_holidays_df(..., country='CN')`)
  - Weekend flag
  - Multivariate forecasting mode
- Offline evaluation (~198 days) showed TimesFM with weather + holidays beating Prophet on MAE/RMSE/MAPE

**Important**: TimesFM-3 weights are subject to the TimesFM Non-Commercial License and **must not be used for production or commercial applications**. Prophet remains the production-safe default.

## 1. Quick Start

### Base Installation (Prophet only)

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### With TimesFM-3 (Optional)

```bash
uv sync --extra timesfm
# or: pip install -e ".[timesfm]"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs:

- http://127.0.0.1:8000/docs

## 2. Project Paths

- Training data folder: `data/` (same level as `app/`, local dev only)
- Default training CSV: `data/historical_flow_from_summary.csv`

For relative paths (for example `data/historical_flow_from_summary.csv`), the API resolves paths against project root.

## 3. API

### `GET /health`

Response example:

```json
{
  "status": "ok",
  "trained": false
}
```

### `POST /train`

Production training from JSON records (used by the orchestrator backend).

```json
{
  "records": [
    {
      "ds": "2026-02-15",
      "y": 4178,
      "temp_max": 28.0,
      "temp_min": 18.0,
      "precip": 0.07,
      "humidity": 68.0,
      "pressure": 1017.0,
      "vis": 25.0,
      "cloud": 68.0,
      "uv_index": 5.0,
      "wind_speed_day": 10.7,
      "wind_speed_night": 9.4
    }
  ],
  "holdout_days": 14,
  "max_training_days": 120,
  "model_name": "multi_weather_regressors"
}
```

Field notes:

- `records`: non-empty array of daily training rows (snake_case fields, date `YYYY-MM-DD`). Do not send derived fields `is_windy_day` or `wind_level`; the service computes them.
- `holdout_days`: validation window for MAE/MAPE evaluation
- `max_training_days`: rolling history window for training (`60` to `720`, default `120`)
- `model_name`: (optional, default `"multi_weather_regressors"`) Model backend to use:
  - `"multi_weather_regressors"` — Prophet (MIT license, production-safe)
  - `"timesfm_weather_holiday"` — TimesFM-3 (non-commercial license, experimental)

Errors:

- `422` if `records` is missing or empty
- `400` if validation, quality gates, or model selection fails

### `POST /train/from-csv`

Local dev training from a CSV file on disk.

```json
{
  "csv_path": "data/historical_flow_from_summary.csv",
  "holdout_days": 14,
  "max_training_days": 120,
  "model_name": "multi_weather_regressors"
}
```

Field notes:

- `csv_path`: historical training CSV path (relative paths resolve against project root)
- `holdout_days`, `max_training_days`: same as `POST /train`
- `model_name`: (optional, default `"multi_weather_regressors"`) same as `POST /train`

Errors:

- `404` if `csv_path` file does not exist
- `400` if validation, quality gates, or model selection fails

Both training endpoints return the same response shape:

Response includes:

- selected model name (`multi_weather_regressors` or `timesfm_weather_holiday`)
- regressors
- training date range and row count
- evaluation metrics

### `POST /predict/next-7-days`

Request body:

```json
{
  "days": 7
}
```

Notes:

- No `future_csv_path` parameter.
- Service fetches QWeather 7-day forecast and uses it directly as model input.
- Set env vars before running:
  - `QWEATHER_API_KEY`
  - `QWEATHER_LOCATION` (QWeather location ID)
- QWeather field semantics:
  - `windSpeedDay` / `windSpeedNight`: km/h
  - `precip`: mm
  - `humidity`: percent (`0-100`)
  - `pressure`: hPa
  - `vis`: km
  - `cloud`: percent (`0-100`), may be nullable from QWeather

### QWeather -> Model Field Mapper

| QWeather field | Model feature |
|---|---|
| `fxDate` | `ds` |
| `tempMax` | `temp_max` |
| `tempMin` | `temp_min` |
| `precip` | `precip` |
| `humidity` | `humidity` |
| `pressure` | `pressure` |
| `vis` | `vis` |
| `cloud` | `cloud` |
| `uvIndex` | `uv_index` |
| `windSpeedDay` | `wind_speed_day` |
| `windSpeedNight` | `wind_speed_night` |

Response example:

```json
{
  "model_name": "multi_weather_regressors",
  "regressors": [
    "temp_max",
    "temp_min",
    "precip",
    "humidity",
    "pressure",
    "vis",
    "cloud",
    "uv_index",
    "wind_speed_day",
    "wind_speed_night"
  ],
  "generated_from": "qweather_7d",
  "predictions": [
    {
      "ds": "2026-03-07",
      "yhat": 3568,
      "yhat_lower": 3210,
      "yhat_upper": 3892
    }
  ]
}
```

**Note**: When using `timesfm_weather_holiday`, the `regressors` field in the response will additionally include `is_holiday` and `is_weekend` (TimesFM uses these as explicit features; Prophet handles holidays internally).

## 4. Orchestrator integration

Recommended daily flow for an external scheduler backend:

1. Query the latest N days of traffic + weather from your data warehouse/API.
2. Map columns to the training schema below (snake_case).
3. `POST /train` with `records`.
4. `POST /predict/next-7-days` (model is in-memory per process; train must run first in the same instance or before predict).
5. Persist prediction results in your system.

The trained model is held in process memory only; restart the service or call `/train` again before `/predict` if the process was recycled.

## 5. Training data schema

Required columns (CSV header or JSON field names):

- `ds`
- `y`
- `temp_max`
- `temp_min`
- `precip`
- `humidity`
- `pressure`
- `vis`
- `cloud`
- `uv_index`
- `wind_speed_day`
- `wind_speed_night`

Legacy compatibility:

- Existing datasets with `weather_score` are still accepted during migration, and missing new fields are backfilled with deterministic defaults.

## 6. Data Quality Gates and Fallbacks

- Forecast ingest enforces core fields (`fxDate`, temp, precip, humidity, pressure, vis, uv, day/night wind).
- `cloud` is treated as nullable; missing values are filled by forecast median (or neutral `50`) before clipping.
- Training data runs quality gates:
  - range validation by QWeather units
  - low-variance detection
  - imputed-like dominance detection (high top-value ratio)
- If critical features (`wind_speed_day`, `vis`, `cloud`) are too low-variance, training fails with an explicit error.
- For low-confidence regressors, Prophet applies lower prior scales to reduce overfitting to synthetic proxies.
