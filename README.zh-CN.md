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
| _(派生)_ | `is_windy_day` | `wind_speed_day ≥ 12 km/h`（蒲福 3 级 / 微风起点）时为 1，否则为 0 |
| _(派生)_ | `wind_level` | 标准蒲福风级整数（0–12），按官方 km/h 边界分箱 |

> **风速派生特征说明**：QWeather 返回的 `windSpeedDay` 在本地区历史数据中分布范围较窄，直接使用原始值方差不足。系统会在训练和预测时自动派生两个特征：
>
> - `is_windy_day`：固定阈值 **12 km/h**（蒲福 3 级微风起点，旌旗展开，游客明显感受到风力），训练与推理使用完全相同的逻辑保持一致。若观测期所有风速均低于阈值（持续静风），则该特征全为 0，模型会自动将其降为低置信度特征（prior scale 降至 0.05）。
> - `wind_level`：采用蒲福风级官方 km/h 边界（见下表），输出 0–12 整数等级。
>
> | 蒲福级 | 中文术语 | 英文 | km/h 范围 |
> |:---:|---|---|---|
> | 0 | 无风 | Calm | < 2 |
> | 1 | 软风 | Light air | 2–5 |
> | 2 | 轻风 | Light breeze | 6–11 |
> | 3 | 微风 | Gentle breeze | 12–19 |
> | 4 | 和风 | Moderate breeze | 20–28 |
> | 5 | 清风 | Fresh breeze | 29–38 |
> | 6 | 强风 | Strong breeze | 39–49 |
> | 7 | 疾风 | Near gale | 50–61 |
> | 8 | 大风 | Gale | 62–74 |
> | 9 | 烈风 | Strong gale | 75–88 |
> | 10 | 狂风 | Storm | 89–102 |
> | 11 | 暴风 | Violent storm | 103–117 |
> | 12 | 飓风 | Hurricane | ≥ 118 |

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
  - `wind_speed_day` 方差过低时**有条件放行**：只要蒲福风级派生特征（`wind_level`）已成功生成，即视为拥有有效风力上下文，训练不会硬失败。原因：持续低风速本身是合法的气象状态（例如静风期），并非数据缺失；模型会通过低置信度路径自动对平坦风速特征降权。
- 对低置信度回归特征（方差过低或单值占比高的特征），Prophet 会自动降低 prior scale（从正常值降至 0.05），减少过拟合风险。
- `is_windy_day` 作为故意设计的二值特征（0/1），不参与低方差判定，但如果正样本比例过高（>85%），也会被降权为低置信度特征。
