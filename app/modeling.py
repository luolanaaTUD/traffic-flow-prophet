from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from prophet import Prophet

MULTI_REGRESSORS = [
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
    # Derived behavioral wind flag — binary (0/1), consistent between training
    # and inference because it uses a fixed absolute threshold (9.5 km/h).
    "is_windy_day",
]


@dataclass
class EvaluationResult:
    model_name: str
    regressors: list[str]
    mae: float | None
    mape: float | None
    holdout_days: int
    status: str


def build_prophet_model(regressors: list[str], low_confidence_regressors: set[str] | None = None) -> Prophet:
    low_confidence_regressors = low_confidence_regressors or set()
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.add_country_holidays(country_name="CN")
    for col in regressors:
        if col in {"temp_max", "temp_min"}:
            prior_scale = 0.2
        elif col in {"precip", "wind_speed_day", "wind_speed_night"}:
            prior_scale = 0.15
        elif col == "is_windy_day":
            # Binary derived flag: conservative prior to avoid overfitting the
            # sparse positive-class (windy days are a minority in training data).
            prior_scale = 0.1
        else:
            prior_scale = 0.1
        if col in low_confidence_regressors:
            prior_scale = min(prior_scale, 0.05)
        model.add_regressor(col, prior_scale=prior_scale)
    return model


def fit_prophet(
    df: pd.DataFrame,
    regressors: list[str],
    low_confidence_regressors: set[str] | None = None,
    use_log_transform: bool = True,
) -> Prophet:
    """
    Fit Prophet model, optionally with log1p transform of target.
    
    Args:
        df: Training dataframe with 'ds', 'y', and regressor columns
        regressors: List of regressor column names
        low_confidence_regressors: Set of regressors to apply lower prior scale
        use_log_transform: If True, train on log1p(y) to prevent negative predictions.
                          The caller must apply expm1 to predictions.
    
    Returns:
        Fitted Prophet model (stores transformed target internally if use_log_transform=True)
    """
    model = build_prophet_model(regressors, low_confidence_regressors=low_confidence_regressors)
    
    train_df = df[["ds", "y"] + regressors].copy()
    if use_log_transform:
        # Transform target to log space to ensure positive predictions
        train_df["y"] = np.log1p(train_df["y"])
    
    model.fit(train_df)
    return model


def evaluate_regressor_set(
    df: pd.DataFrame,
    model_name: str,
    regressors: list[str],
    holdout_days: int,
    low_confidence_regressors: set[str] | None = None,
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
    use_log_transform = True  # Use same transform as production model
    
    model = fit_prophet(
        train_df,
        regressors,
        low_confidence_regressors=low_confidence_regressors,
        use_log_transform=use_log_transform,
    )
    pred = model.predict(valid_df[["ds"] + regressors])[["ds", "yhat"]]
    
    # Apply inverse transform if needed
    if use_log_transform:
        pred["yhat"] = np.expm1(pred["yhat"])
    
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
    low_confidence_regressors: set[str] | None = None,
) -> tuple[Prophet, str, list[str], list[EvaluationResult], bool]:
    """
    Train Prophet model with weather regressors and CN holidays.
    
    Returns:
        (model, model_name, regressors, evaluation_results, use_log_transform)
        The use_log_transform flag indicates whether predictions need expm1 inverse transform.
    """
    selected_model_name = "multi_weather_regressors"
    selected_regressors = MULTI_REGRESSORS
    use_log_transform = True  # Enable log transform to prevent negative predictions

    evaluation = evaluate_regressor_set(
        df=df,
        model_name=selected_model_name,
        regressors=selected_regressors,
        holdout_days=holdout_days,
        low_confidence_regressors=low_confidence_regressors,
    )
    evaluation_results = [evaluation]

    # Train final model on full history window using fixed multi-regressor feature set.
    model = fit_prophet(
        df,
        selected_regressors,
        low_confidence_regressors=low_confidence_regressors,
        use_log_transform=use_log_transform,
    )
    return model, selected_model_name, selected_regressors, evaluation_results, use_log_transform
