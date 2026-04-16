# Traffic Flow Prophet Backend

Chinese version: [README.zh-CN.md](README.zh-CN.md)

This backend is extracted from the notebook workflow and provides:

- model training from historical traffic CSV
- next-7-days traffic prediction API

The training model is fixed to `multi_weather_regressors` (no baseline switch).

## 1. Quick Start

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs:

- http://127.0.0.1:8000/docs

## 2. Project Paths

- Training data folder: `data/` (same level as `app/`)
- Default training CSV: `data/historical_flow.csv`

For relative paths (for example `data/historical_flow.csv`), the API resolves paths against project root.

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

Request body:

```json
{
  "csv_path": "data/historical_flow.csv",
  "holdout_days": 14,
  "max_training_days": 1200
}
```

Field notes:

- `csv_path`: historical training CSV path
- `holdout_days`: validation window for MAE/MAPE evaluation
- `max_training_days`: rolling history window for training (`60` to `1200`)

Response includes:

- selected model name (`multi_weather_regressors`)
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

## 4. Training CSV Schema

Required columns:

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

## 5. Data Quality Gates and Fallbacks

- Forecast ingest enforces core fields (`fxDate`, temp, precip, humidity, pressure, vis, uv, day/night wind).
- `cloud` is treated as nullable; missing values are filled by forecast median (or neutral `50`) before clipping.
- Training data runs quality gates:
  - range validation by QWeather units
  - low-variance detection
  - imputed-like dominance detection (high top-value ratio)
- If critical features (`wind_speed_day`, `vis`, `cloud`) are too low-variance, training fails with an explicit error.
- For low-confidence regressors, Prophet applies lower prior scales to reduce overfitting to synthetic proxies.
