# Traffic Flow Prophet 后端服务

英文版文档: [README.md](README.md)

本后端服务从 Notebook 工作流抽取而来，提供：

- 基于历史客流 CSV 的模型训练
- 未来 7 天客流预测 API

训练模型固定为 `multi_weather_regressors`（不再做 baseline 切换）。

## 1. 快速开始

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger 文档：

- http://127.0.0.1:8000/docs

## 2. 项目路径约定

- 训练数据目录：`data/`（与 `app/` 同级）
- 默认训练文件：`data/historical_flow.csv`

对于相对路径（例如 `data/historical_flow.csv`），API 会按项目根目录解析。

## 3. API

### `GET /health`

返回示例：

```json
{
  "status": "ok",
  "trained": false
}
```

### `POST /train`

请求体：

```json
{
  "csv_path": "data/historical_flow.csv",
  "holdout_days": 14,
  "max_training_days": 1095
}
```

字段说明：

- `csv_path`：历史训练 CSV 路径
- `holdout_days`：用于 MAE/MAPE 评估的验证窗口天数
- `max_training_days`：用于训练的最大历史窗口（最多 3 年，即 `1095`）

响应包含：

- 模型名称（`multi_weather_regressors`）
- 回归特征列表
- 训练数据行数与日期范围
- 评估指标

### `POST /predict/next-7-days`

请求体：

```json
{
  "days": 7
}
```

说明：

- 不再支持 `future_csv_path` 参数。
- 未来特征由服务根据历史画像自动生成。

返回示例：

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

## 4. 训练 CSV 字段要求

必填字段：

- `ds`
- `y`
- `temp_max`
- `weather_score`

可选字段（建议提供以提升多回归模型效果）：

- `precip`
- `wind_speed_day`
- `humidity`
- `uv_index`
- `is_rain`
- `is_severe_weather`

如果可选天气字段缺失，服务会基于已有字段自动补全与增强。
