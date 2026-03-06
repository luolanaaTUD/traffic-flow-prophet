# 海边公园客流 AI 预测模型开发文档

## 1. 业务目标
基于历史 3 个月的真实客流数据，结合**中国法定节假日**与**未来 7 天天气预报**，使用时间序列模型（Facebook Prophet）每日动态预测未来 7 天的入园总人次，为前端大屏提供平滑的预测折线图。

## 2. 数据准备与特征工程
Prophet 模型对数据格式有严格要求，核心列必须命名为 `ds` (日期) 和 `y` (目标值)。

### 2.1 训练集数据结构 (合并历史 CSV 与 历史天气)
| ds (日期) | y (真实入园人次) | temp_max (最高温) | weather_score (天气降权系数) |
| :--- | :--- | :--- | :--- |
| 2023-09-01 | 4500 | 28 | 1.0 (晴天) |
| 2023-09-02 | 1200 | 25 | 0.3 (大雨) |

> **💡 天气降权系数 (`weather_score`) 映射建议：**
> Prophet 的附加回归项需要**数值型特征**。建议将外部天气 API 返回的文本转换为系数：
> *   晴天 / 多云：`1.0`
> *   阴天：`0.8`
> *   小雨 / 小雪：`0.5`
> *   中到大雨 / 极端天气：`0.2`

### 2.2 预测集数据结构 (构建未来 7 天特征)
包含未来 7 天的连续日期 `ds`，以及通过天气 API 获取的未来 7 天 `temp_max` 和 `weather_score`。

## 3. 模型设计与参数配置
*   **基础模型**：`prophet.Prophet`
*   **周期性设置**：
    *   `weekly_seasonality=True` (强依赖：公园周末客流显著高于工作日)
    *   `yearly_seasonality=False` (注意：由于只有3个月数据，开启年度周期会导致过拟合报错，需关闭)
*   **节假日效应**：使用内置的 `add_country_holidays(country_name='CN')` 自动捕获中国法定节假日及调休。
*   **外部变量 (Regressors)**：添加最高温和天气系数作为外部特征。

## 4. 核心代码实现 (Python)
以下为可直接在服务器或任务脚本中运行的核心代码结构：

```python
import pandas as pd
from prophet import Prophet
import requests
import datetime
import pymysql # 或其他数据库驱动

# ==========================================
# 步骤 1: 数据加载与预处理
# ==========================================
def load_historical_data():
    # 假设 csv 已经包含了 ds, y, temp_max, weather_score
    df = pd.read_csv('historical_flow.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# ==========================================
# 步骤 2: 模型构建与训练
# ==========================================
def train_prophet_model(df_train):
    # 初始化模型，关闭年度周期(数据量不足一年)，开启周周期
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True)
    
    # 1. 载入中国法定节假日（包含春节、五一、十一等及调休）
    m.add_country_holidays(country_name='CN')
    
    # 2. 添加天气和气温作为外部回归特征
    m.add_regressor('temp_max')
    m.add_regressor('weather_score')
    
    # 3. 拟合历史数据
    m.fit(df_train)
    return m

# ==========================================
# 步骤 3: 获取未来7天特征 (伪代码)
# ==========================================
def get_future_weather_features():
    # 调用第三方天气预报 API (如和风天气)
    # response = requests.get("天气API_URL").json()
    # 解析出未来 7 天的日期、最高温、天气状态并转化为 weather_score
    
    # 模拟生成未来 7 天的特征 DataFrame
    today = datetime.date.today()
    future_dates =[today + datetime.timedelta(days=i) for i in range(1, 8)]
    
    future_df = pd.DataFrame({
        'ds': future_dates,
        'temp_max':[22, 24, 25, 20, 18, 22, 23], # 模拟气温
        'weather_score':[1.0, 1.0, 0.8, 0.5, 0.2, 1.0, 1.0] # 模拟天气系数
    })
    return future_df

# ==========================================
# 步骤 4: 执行预测与后处理
# ==========================================
def run_prediction_job():
    # 1. 训练模型
    df_train = load_historical_data()
    model = train_prophet_model(df_train)
    
    # 2. 准备未来 7 天的基准数据框
    future_df = get_future_weather_features()
    
    # 3. 执行预测
    forecast = model.predict(future_df)
    
    # 4. 提取核心预测结果: ds (日期) 和 yhat (预测值)
    results = forecast[['ds', 'yhat']].copy()
    
    # 【核心防御逻辑】：Prophet 可能会预测出负数，客流必须截断为 0
    results['yhat'] = results['yhat'].apply(lambda x: int(max(0, x)))
    
    # 5. 保存结果到数据库
    save_to_database(results)
    print("预测完成，结果已写入数据库。")

def save_to_database(results_df):
    # 遍历 results_df，将未来的日期和预测值写入之前设计的 flow_prediction_daily 表
    # SQL 采用 INSERT INTO ... ON DUPLICATE KEY UPDATE 确保数据幂等性
    pass

if __name__ == "__main__":
    run_prediction_job()
```

## 5. 调度与部署方案
推荐使用轻量级的 Linux 定时任务（**Crontab**）来实现“每天晚上触发”。

1.  **服务器环境**：Python 3.8+，安装依赖 `pip install pandas prophet requests pymysql`
2.  **脚本部署**：将上述代码保存为 `predict_job.py`。
3.  **配置 Crontab**：在服务器终端输入 `crontab -e`，添加以下记录：
    ```bash
    # 每天凌晨 01:30 执行一次模型训练与预测，并将日志输出
    30 1 * * * /usr/bin/python3 /path/to/predict_job.py >> /var/log/park_predict.log 2>&1
    ```

## 6. 边缘异常处理 (Edge Cases)
在实际生产中，为了保证大屏不出现“空图”或“离谱的断崖线”，模型需加入以下兜底策略：
1.  **天气 API 挂了怎么办？**
    如果请求第三方天气失败，`temp_max` 取历史本月平均气温，`weather_score` 默认取 `1.0` (默认非极端天气)。
2.  **新疫情或极端封园情况（重大突变）**
    由于 Prophet 是基于历史平滑拟合的，无法预测突发的行政封园指令。预测结果如果与实际严重不符，应在后台系统提供**“人工干预干预预测值”**的覆盖接口（Override）。前端大屏优先读取人工干预值，无干预值时读取模型预测值。