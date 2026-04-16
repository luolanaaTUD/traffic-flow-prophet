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
  "max_training_days": 1200
}
```

字段说明：

- `csv_path`：历史训练 CSV 路径
- `holdout_days`：用于 MAE/MAPE 评估的验证窗口天数
- `max_training_days`：训练滚动窗口天数（范围 `60` 到 `1200`）

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
- 服务会实时拉取 QWeather 未来 7 天天气并直接作为模型输入。
- 启动前需设置环境变量：
  - `QWEATHER_API_KEY`
  - `QWEATHER_LOCATION`（QWeather 的 location ID）
- QWeather 字段语义：
  - `windSpeedDay` / `windSpeedNight`：单位 km/h
  - `precip`：单位 mm
  - `humidity`：百分比（`0-100`）
  - `pressure`：单位 hPa
  - `vis`：单位 km
  - `cloud`：百分比（`0-100`），QWeather 可能返回空值

### QWeather -> 模型字段映射

| QWeather 字段 | 模型特征字段 |
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

返回示例：

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

## 4. 训练 CSV 字段要求

必填字段：

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

兼容说明：

- 为兼容存量数据，若 CSV 含 `weather_score` 且缺少新字段，服务会在迁移期按规则补齐。

## 5. 数据质量门禁与回退策略

- 预测侧会强校验核心字段（`fxDate`、温度、降水、湿度、气压、能见度、UV、昼夜风速）。
- `cloud` 允许为空；为空时会使用预测窗口中位数（若全空则用中性值 `50`）再做裁剪。
- 训练前会执行数据质量门禁：
  - 按 QWeather 单位做范围校验
  - 低方差特征检测
  - 疑似插补主导检测（单值占比过高）
- 若关键特征（`wind_speed_day`、`vis`、`cloud`）方差过低，将直接报错阻止训练。
- 对低置信度回归特征，Prophet 会自动降低 prior scale，减少对合成代理特征的过拟合风险。
