# Traffic Flow Prophet Backend

Backend API built from the notebook logic. It supports:

- training fixed Prophet `multi_weather_regressors` model from historical traffic CSV
- predicting next 7 days traffic

## 1. Run with uv

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

- http://127.0.0.1:8000/docs

## 2. Data format

Training CSV must include:

- `ds` (date)
- `y` (traffic count)
- `temp_max` (max temperature)
- `weather_score` (weather impact score)

Training window:

- `max_training_days` defaults to `1095` (up to 3 years of recent history)

Optional columns supported by the multi-regressor model:

- `precip`
- `wind_speed_day`
- `humidity`
- `uv_index`
- `is_rain`
- `is_severe_weather`

## 3. API examples

### Train model from CSV

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/historical_flow.csv",
    "holdout_days": 14,
    "max_training_days": 1095
  }'
```

### Predict next 7 days

Use features inferred from historical profile:

```bash
curl -X POST http://127.0.0.1:8000/predict/next-7-days \
  -H "Content-Type: application/json" \
  -d '{
    "days": 7
  }'
```

Or provide your own future feature CSV:

```bash
curl -X POST http://127.0.0.1:8000/predict/next-7-days \
  -H "Content-Type: application/json" \
  -d '{
    "days": 7,
    "future_csv_path": "data/future_features.csv"
  }'
```
