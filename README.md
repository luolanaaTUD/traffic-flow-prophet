# Traffic Flow Prophet Backend

Chinese version: [README.zh-CN.md](README.zh-CN.md)

This backend is extracted from the notebook workflow and provides:

- model training from JSON records (production) or historical traffic CSV (local dev)
- next-7-days traffic prediction API
- **TimesFM-3 forecasting as default backend** with Prophet available as opt-in

The default training model is `timesfm_weather_holiday` (TimesFM-3). You can optionally select `multi_weather_regressors` (Prophet) via the `model_name` parameter.

⚠️ **IMPORTANT LICENSE NOTICE**: The default TimesFM-3 model is subject to the **TimesFM Non-Commercial License** and **must NOT be used for production or commercial applications**. For production use, explicitly select Prophet by passing `"model_name": "multi_weather_regressors"`.

## Model Selection

### TimesFM-3 (Default, Non-Commercial)
- Model name: `timesfm_weather_holiday` (DEFAULT)
- Checkpoint: `google/timesfm-3.0-pytorch`
- License: **TimesFM Non-Commercial License** (non-commercial / non-production use only)
- Requires: `pip install -e ".[timesfm]"` or `pip install timesfm[torch] torch>=2.0.0`
- Features:
  - Weather covariates (temp_max, temp_min, precip, humidity, pressure, vis, cloud, uv_index, wind_speed_day, wind_speed_night, is_windy_day)
  - CN holidays from Prophet's calendar (`make_holidays_df(..., country='CN')`)
  - Weekend flag
  - Multivariate forecasting mode
- Offline evaluation (~198 days) showed TimesFM with weather + holidays beating Prophet on MAE/RMSE/MAPE

### Prophet (Production-Safe Opt-In)
- Model name: `multi_weather_regressors`
- License: MIT (production-safe)
- Baseline model for traffic forecasting
- To use Prophet, explicitly pass `"model_name": "multi_weather_regressors"` in training requests

**⚠️ WARNING**: If you omit `model_name`, the service uses TimesFM by default, which requires the optional `timesfm` package. If not installed, training will fail with an explicit error. Install TimesFM dependencies or explicitly specify Prophet for production use.

## 1. Quick Start

### With TimesFM-3 (Default, requires optional dependency)

```bash
uv sync --extra timesfm
# or: pip install -e ".[timesfm]"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Prophet-only (Production-safe, lighter install)

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Note**: If TimesFM dependencies are not installed, training requests will fail unless you explicitly specify `"model_name": "multi_weather_regressors"` to use Prophet.

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

**Default behavior (TimesFM-3, non-commercial license):**

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
  "max_training_days": 120
}
```

**Production-safe opt-in (Prophet, MIT license):**

```json
{
  "records": [ /* ... */ ],
  "holdout_days": 14,
  "max_training_days": 120,
  "model_name": "multi_weather_regressors"
}
```

Field notes:

- `records`: non-empty array of daily training rows (snake_case fields, date `YYYY-MM-DD`). Do not send derived fields `is_windy_day` or `wind_level`; the service computes them.
- `holdout_days`: validation window for MAE/MAPE evaluation
- `max_training_days`: rolling history window for training (`60` to `720`, default `120`)
- `model_name`: (optional, default `"timesfm_weather_holiday"`) Model backend to use:
  - `"timesfm_weather_holiday"` — TimesFM-3 (DEFAULT, non-commercial license, requires timesfm package)
  - `"multi_weather_regressors"` — Prophet (MIT license, production-safe)

⚠️ **License Warning**: The default TimesFM-3 model is non-commercial. For production use, explicitly pass `"model_name": "multi_weather_regressors"`.

Errors:

- `422` if `records` is missing or empty
- `400` if validation, quality gates, or model selection fails
- `500` with ImportError if TimesFM requested but not installed (install hint included)

### `POST /train/from-csv`

Local dev training from a CSV file on disk.

**Default (TimesFM-3):**

```json
{
  "csv_path": "data/historical_flow_from_summary.csv",
  "holdout_days": 14,
  "max_training_days": 120
}
```

**Prophet opt-in:**

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
- `model_name`: (optional, default `"timesfm_weather_holiday"`) same as `POST /train`

Errors:

- `404` if `csv_path` file does not exist
- `400` if validation, quality gates, or model selection fails
- `500` with ImportError if TimesFM requested but not installed

Both training endpoints return the same response shape:

Response includes:

- selected model name (`timesfm_weather_holiday` or `multi_weather_regressors`)
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
  "model_name": "timesfm_weather_holiday",
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
    "wind_speed_night",
    "is_windy_day",
    "is_holiday",
    "is_weekend"
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

**Note**: 
- TimesFM (default) includes `is_holiday` and `is_weekend` in regressors
- Prophet includes only weather regressors (holidays handled internally)
- Response format matches whichever model was used during training

## 4. Orchestrator integration

Recommended daily flow for an external scheduler backend:

1. Query the latest N days of traffic + weather from your data warehouse/API.
2. Map columns to the training schema below (snake_case).
3. `POST /train` with `records`.
   - **Default**: Uses TimesFM-3 (non-commercial license) unless `model_name` specified
   - **Production**: Pass `"model_name": "multi_weather_regressors"` for Prophet (MIT license)
4. `POST /predict/next-7-days` (model is in-memory per process; train must run first in the same instance or before predict).
5. Persist prediction results in your system.

The trained model is held in process memory only; restart the service or call `/train` again before `/predict` if the process was recycled.

⚠️ **For production deployments**: Explicitly pass `"model_name": "multi_weather_regressors"` in all training requests to use the MIT-licensed Prophet model. The default TimesFM-3 model is subject to non-commercial license restrictions.

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

## 7. Testing TimesFM-3 (Optional Backend)

Since TimesFM-3 requires downloading large model weights (~2GB+), automated CI tests run Prophet only. To manually test TimesFM:

### Setup

```bash
# Install TimesFM dependencies
pip install -e ".[timesfm]"
# or: pip install timesfm[torch] torch>=2.0.0
```

### Manual Testing

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Train with TimesFM
curl -X POST http://localhost:8000/train/from-csv \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/historical_flow_from_summary.csv",
    "holdout_days": 14,
    "max_training_days": 120,
    "model_name": "timesfm_weather_holiday"
  }'

# Predict (ensure QWeather env vars are set)
export QWEATHER_API_KEY="your_key"
export QWEATHER_LOCATION="your_location_id"

curl -X POST http://localhost:8000/predict/next-7-days \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

**Expected behavior:**
- Training response includes `"model_name": "timesfm_weather_holiday"`
- Regressors list includes `is_holiday` and `is_weekend` (in addition to weather features)
- Predictions return 7-day forecast with `yhat`, `yhat_lower`, `yhat_upper`

**First run:** TimesFM will download checkpoint weights from HuggingFace (`google/timesfm-3.0-pytorch`). This may take several minutes depending on network speed.
