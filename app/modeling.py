from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from prophet import Prophet

MULTI_REGRESSORS = [
    "temp_max",
    "precip",
    "wind_speed_day",
    "humidity",
    "uv_index",
    "is_rain",
    "is_severe_weather",
]


@dataclass
class EvaluationResult:
    model_name: str
    regressors: list[str]
    mae: float | None
    mape: float | None
    holdout_days: int
    status: str


def build_prophet_model(regressors: list[str]) -> Prophet:
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.add_country_holidays(country_name="CN")
    for col in regressors:
        prior_scale = 0.2 if col == "temp_max" else 0.1
        model.add_regressor(col, prior_scale=prior_scale)
    return model


def fit_prophet(df: pd.DataFrame, regressors: list[str]) -> Prophet:
    model = build_prophet_model(regressors)
    model.fit(df[["ds", "y"] + regressors].copy())
    return model


def evaluate_regressor_set(
    df: pd.DataFrame,
    model_name: str,
    regressors: list[str],
    holdout_days: int,
) -> EvaluationResult:
    missing = [col for col in regressors if col not in df.columns]
    if missing:
        return EvaluationResult(
            model_name=model_name,
            regressors=regressors,
            mae=None,
            mape=None,
            holdout_days=0,
            status=f"missing columns: {missing}",
        )

    clamped_holdout = max(7, min(holdout_days, len(df) // 4))
    if len(df) < 45 or clamped_holdout < 7:
        return EvaluationResult(
            model_name=model_name,
            regressors=regressors,
            mae=None,
            mape=None,
            holdout_days=0,
            status="trained_without_holdout",
        )

    train_df = df.iloc[:-clamped_holdout].copy()
    valid_df = df.iloc[-clamped_holdout:].copy()
    model = fit_prophet(train_df, regressors)
    pred = model.predict(valid_df[["ds"] + regressors])[["ds", "yhat"]]
    merged = valid_df[["ds", "y"]].merge(pred, on="ds", how="left")

    abs_err = (merged["y"] - merged["yhat"]).abs()
    mae = float(abs_err.mean())
    mape = float((abs_err / merged["y"].clip(lower=1)).mean() * 100)

    return EvaluationResult(
        model_name=model_name,
        regressors=regressors,
        mae=mae,
        mape=mape,
        holdout_days=int(clamped_holdout),
        status="evaluated",
    )


def train_multi_model(
    df: pd.DataFrame,
    holdout_days: int = 14,
) -> tuple[Prophet, str, list[str], list[EvaluationResult]]:
    selected_model_name = "multi_weather_regressors"
    selected_regressors = MULTI_REGRESSORS

    evaluation = evaluate_regressor_set(
        df=df,
        model_name=selected_model_name,
        regressors=selected_regressors,
        holdout_days=holdout_days,
    )
    evaluation_results = [evaluation]

    # Train final model on full history window using fixed multi-regressor feature set.
    model = fit_prophet(df, selected_regressors)
    return model, selected_model_name, selected_regressors, evaluation_results
