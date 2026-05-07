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

| QWeather 字段 | 模型特征字段 | 说明 |
|---|---|---|
| `fxDate` | `ds` | |
| `tempMax` | `temp_max` | |
| `tempMin` | `temp_min` | |
| `precip` | `precip` | |
| `humidity` | `humidity` | |
| `pressure` | `pressure` | |
| `vis` | `vis` | |
| `cloud` | `cloud` | |
| `uvIndex` | `uv_index` | |
| `windSpeedDay` | `wind_speed_day` | 原始值；同时派生下方两个特征 |
| `windSpeedNight` | `wind_speed_night` | |
| _(派生)_ | `is_windy_day` | `wind_speed_day > 9.5 km/h` 时为 1，否则为 0 |
| _(派生)_ | `wind_level` | Beaufort 近似等级：0=静风(<5.5)，1=轻风(5.5–11.5)，2=微风(11.5–19.5)，3=和风(19.5–28.5)，4=大风(≥28.5) |

> **风速派生特征说明**：QWeather 返回的 `windSpeedDay` 在本地区历史数据中分布范围较窄（常见值仅 2–3 个量级），直接使用原始值作为回归特征方差不足。系统会在训练和预测时自动从原始风速派生 `is_windy_day`（固定阈值 9.5 km/h，对应 QWeather 风力 2–3 级分界）和 `wind_level`（Beaufort 等级分箱）。阈值固定可确保训练与推理的派生逻辑完全一致。

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
    "wind_speed_night",
    "is_windy_day"
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
  - 低方差特征检测（含 `is_windy_day`、`wind_level` 派生特征）
  - 疑似插补主导检测（单值占比过高）
- 硬失败规则（会阻止训练）：
  - `vis`（能见度）或 `cloud`（云量）方差过低时直接报错。
  - `wind_speed_day` 方差过低时**有条件放行**：若派生特征 `is_windy_day` 存在且有正样本（即有至少 1 天风速超过 9.5 km/h），则不硬失败。原因：QWeather 在本地区返回的风速精度有限，数值常集中于 2–3 个量级，但派生的二值特征仍能保留对游客行为有影响的风力信号。
- 对低置信度回归特征（方差过低或单值占比高的特征），Prophet 会自动降低 prior scale（从正常值降至 0.05），减少过拟合风险。
- `is_windy_day` 作为故意设计的二值特征（0/1），不参与低方差判定，但如果正样本比例过高（>85%），也会被降权为低置信度特征。
