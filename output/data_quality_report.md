# Data Quality Audit

- Source: `data/historical_flow_from_summary.csv`
- Rows: `60`
- Date range: `2026-02-15` to `2026-04-15`

## Feature Stats
| feature | nunique | unique_ratio | top_value_ratio |
|---|---:|---:|---:|
| temp_max | 12 | 0.2000 | 0.1500 |
| temp_min | 11 | 0.1833 | 0.1500 |
| precip | 26 | 0.4333 | 0.5667 |
| humidity | 17 | 0.2833 | 0.1500 |
| pressure | 25 | 0.4167 | 0.1500 |
| vis | 15 | 0.2500 | 0.7167 |
| cloud | 28 | 0.4667 | 0.1500 |
| uv_index | 6 | 0.1000 | 0.3333 |
| wind_speed_day | 2 | 0.0333 | 0.6833 |
| wind_speed_night | 2 | 0.0333 | 0.6833 |

## Correlation With y
| feature | corr_with_y |
|---|---:|
| uv_index | 0.2513 |
| vis | 0.2143 |
| wind_speed_night | -0.0077 |
| wind_speed_day | -0.0077 |
| temp_min | -0.0685 |
| temp_max | -0.0815 |
| pressure | -0.1337 |
| precip | -0.2147 |
| humidity | -0.2870 |
| cloud | -0.2898 |

## Quality Gate Result
- Status: `FAIL`
- Reason: `Training weather features are too low-variance for reliable learning: ['wind_speed_day']. Please provide richer observed weather values.`