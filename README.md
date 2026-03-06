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
  "max_training_days": 1095
}
```

Field notes:

- `csv_path`: historical training CSV path
- `holdout_days`: validation window for MAE/MAPE evaluation
- `max_training_days`: max history window used for training (up to 3 years = `1095`)

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
- Future features are generated internally from historical profile.

Response example:

```json
{
  "model_name": "multi_weather_regressors",
  "regressors": [
    "temp_max",
    "precip",
    "wind_speed_day",
    "humidity",
    "uv_index",
    "is_rain",
    "is_severe_weather"
  ],
  "generated_from": "historical_profile",
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
- `weather_score`

Optional columns (recommended for full multi-regressor quality):

- `precip`
- `wind_speed_day`
- `humidity`
- `uv_index`
- `is_rain`
- `is_severe_weather`

If optional weather columns are missing, the service derives reasonable defaults/enrichment from existing fields.
