from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from prophet.make_holidays import make_holidays_df

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


def _try_import_timesfm():
    """Lazy import TimesFM-3 to avoid hard dependency on torch."""
    try:
        from timesfm3 import ModelConfig, TimesFM3Evaluator
        return ModelConfig, TimesFM3Evaluator
    except ImportError:
        try:
            # Fallback: try the alternate import path
            import timesfm
            return timesfm.ModelConfig, timesfm.TimesFM3Evaluator
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "TimesFM-3 is not installed. Install with: pip install timesfm[torch] torch>=2.0.0"
            ) from exc


def _make_cn_holiday_features(ds_col: pd.Series) -> pd.DataFrame:
    """
    Generate CN holiday features matching Prophet's add_country_holidays(country_name='CN').
    
    Returns DataFrame with columns:
    - ds: datetime
    - is_holiday: 1 if holiday, 0 otherwise
    - is_weekend: 1 if Saturday/Sunday, 0 otherwise
    """
    min_date = ds_col.min()
    max_date = ds_col.max()
    
    # Use Prophet's CN holiday calendar (same source as Prophet model)
    cn_holidays_df = make_holidays_df(
        year_list=list(range(min_date.year - 1, max_date.year + 2)),
        country='CN'
    )
    
    # Create feature DataFrame
    out = pd.DataFrame({'ds': ds_col})
    out['ds_date'] = out['ds'].dt.normalize()
    
    # Mark holidays
    cn_holidays_df['ds'] = pd.to_datetime(cn_holidays_df['ds']).dt.normalize()
    holiday_dates = set(cn_holidays_df['ds'].values)
    out['is_holiday'] = out['ds_date'].isin(holiday_dates).astype(float)
    
    # Mark weekends
    out['is_weekend'] = out['ds'].dt.dayofweek.isin([5, 6]).astype(float)
    
    return out[['ds', 'is_holiday', 'is_weekend']]


def _prepare_timesfm_covariates(
    df: pd.DataFrame,
    regressors: list[str],
    include_holidays: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Prepare covariates for TimesFM-3 multivariate mode.
    
    Args:
        df: DataFrame with 'ds' column and weather regressors
        regressors: List of weather regressor column names
        include_holidays: If True, add CN holiday + weekend features
    
    Returns:
        (covariates_array, covariate_names) where covariates_array has shape (num_features, n_samples)
    """
    covariate_cols = regressors.copy()
    cov_df = df[regressors].copy()
    
    if include_holidays:
        holiday_features = _make_cn_holiday_features(df['ds'])
        cov_df = cov_df.join(holiday_features[['is_holiday', 'is_weekend']])
        covariate_cols.extend(['is_holiday', 'is_weekend'])
    
    # Convert to numpy array: shape (n_samples, n_features), then transpose to (n_features, n_samples)
    # TimesFM-3 expects (C, T) shape for covariates
    covariates = cov_df.values.T.astype(np.float32)
    
    return covariates, covariate_cols


def _normalize_covariates(
    covariates: np.ndarray,
    train_mean: np.ndarray | None = None,
    train_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize covariates using training statistics.
    
    Args:
        covariates: Shape (C, T) or (C, T+H)
        train_mean: Training mean per feature (C,). If None, computed from covariates.
        train_std: Training std per feature (C,). If None, computed from covariates.
    
    Returns:
        (normalized_covariates, mean, std)
    """
    if train_mean is None:
        train_mean = covariates.mean(axis=1, keepdims=True)
    else:
        train_mean = train_mean.reshape(-1, 1)
    
    if train_std is None:
        train_std = covariates.std(axis=1, keepdims=True)
    else:
        train_std = train_std.reshape(-1, 1)
    
    # Avoid division by zero
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    
    normalized = (covariates - train_mean) / train_std
    
    return normalized, train_mean.flatten(), train_std.flatten()


def fit_timesfm(
    df: pd.DataFrame,
    regressors: list[str],
    checkpoint_path: str = "google/timesfm-3.0-pytorch",
    low_confidence_regressors: set[str] | None = None,
) -> tuple[object, list[str]]:
    """
    Fit TimesFM-3 model with weather covariates and CN holidays.
    
    Note: TimesFM-3 is a pretrained foundation model. This "fit" method 
    prepares training context and covariates for zero-shot forecasting.
    
    Args:
        df: Training DataFrame with 'ds', 'y', and weather regressors
        regressors: Weather regressor column names
        checkpoint_path: TimesFM-3 model checkpoint
        low_confidence_regressors: (not used for TimesFM; kept for API compatibility)
    
    Returns:
        (model, covariate_names)
    """
    ModelConfig, TimesFM3Evaluator = _try_import_timesfm()
    
    # Initialize TimesFM-3 model with proper configuration
    device = "cuda" if _has_cuda() else "cpu"
    config = ModelConfig(
        checkpoint_path=checkpoint_path,
        per_core_batch_size=1,
        device=device,
    )
    model = TimesFM3Evaluator(config)
    
    # Prepare covariates (weather + holidays + weekend)
    # Shape: (num_features, time_steps)
    covariates, covariate_names = _prepare_timesfm_covariates(
        df=df,
        regressors=regressors,
        include_holidays=True,
    )
    
    # Normalize covariates using training statistics
    normalized_covariates, train_mean, train_std = _normalize_covariates(covariates)
    
    # TimesFM-3 expects time series context
    # Shape: (1, time_steps) for univariate target
    y_values = df['y'].values.astype(np.float32).reshape(1, -1)
    
    # Store training context for later prediction
    # TimesFM-3 is a foundation model and doesn't need explicit training
    model._training_data = {
        'y': y_values,
        'covariates': normalized_covariates,  # (C, T)
        'covariate_names': covariate_names,
        'train_mean': train_mean,
        'train_std': train_std,
        'last_ds': df['ds'].iloc[-1],
        'freq': 'D',
    }
    
    return model, covariate_names


def predict_timesfm(
    model: object,
    future_df: pd.DataFrame,
    days: int = 7,
) -> pd.DataFrame:
    """
    Generate predictions using TimesFM-3 foundation model.
    
    Args:
        model: TimesFM-3 model (with _training_data attached)
        future_df: Future DataFrame with 'ds' and weather regressors
        days: Number of days to predict
    
    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    training_data = model._training_data
    covariate_names = training_data['covariate_names']
    
    # Extract weather regressors from covariate_names (exclude holiday/weekend)
    weather_regressors = [col for col in covariate_names if col not in {'is_holiday', 'is_weekend'}]
    
    # Prepare future covariates (C, H)
    future_covariates, _ = _prepare_timesfm_covariates(
        df=future_df.head(days),
        regressors=weather_regressors,
        include_holidays=True,
    )
    
    # Normalize future covariates using training statistics
    future_covariates_norm = (future_covariates - training_data['train_mean'].reshape(-1, 1)) / training_data['train_std'].reshape(-1, 1)
    
    # Combine past and future covariates: (C, T+H)
    past_covariates = training_data['covariates']  # (C, T)
    combined_covariates = np.concatenate([past_covariates, future_covariates_norm], axis=1)  # (C, T+H)
    
    # Context target: (1, T)
    context_target = training_data['y']  # Already (1, T)
    
    # Make prediction with TimesFM-3 API
    # predict_batch expects:
    # - context: (batch_size, context_len)
    # - past_future_covariates: (num_features, context_len + horizon_len)
    # - horizon_len: int
    # - univariate: False to use covariates
    # - return_quantiles: True to get uncertainty bands
    
    try:
        forecast_result = model.predict_batch(
            context=context_target,  # (1, T)
            past_future_covariates=combined_covariates,  # (C, T+H)
            horizon_len=days,
            univariate=False,  # CRITICAL: Must be False to use covariates
            return_quantiles=True,  # Get quantiles for uncertainty
        )
        
        # forecast_result shape: (batch_size, horizon, num_quantiles)
        # Typically quantiles are [0.1, 0.5, 0.9]
        if isinstance(forecast_result, dict):
            # Handle dict return format
            if 'quantiles' in forecast_result:
                quantiles = forecast_result['quantiles']  # (1, H, Q)
                point_forecast = quantiles[0, :, 1]  # Median (0.5 quantile)
                lower_forecast = quantiles[0, :, 0]  # 10th percentile
                upper_forecast = quantiles[0, :, 2]  # 90th percentile
            elif 'point_forecast' in forecast_result:
                point_forecast = forecast_result['point_forecast'][0, :]  # (H,)
                # Generate approximate uncertainty bands (±20%)
                lower_forecast = point_forecast * 0.8
                upper_forecast = point_forecast * 1.2
            else:
                raise ValueError(f"Unexpected forecast_result format: {forecast_result.keys()}")
        else:
            # Handle array return format
            if len(forecast_result.shape) == 3:
                # (batch, horizon, quantiles)
                point_forecast = forecast_result[0, :, 1]  # Median
                lower_forecast = forecast_result[0, :, 0]  # 10th percentile
                upper_forecast = forecast_result[0, :, 2]  # 90th percentile
            else:
                # (batch, horizon) - point forecast only
                point_forecast = forecast_result[0, :]
                # Generate approximate uncertainty bands (±20%)
                lower_forecast = point_forecast * 0.8
                upper_forecast = point_forecast * 1.2
    
    except Exception as exc:
        raise RuntimeError(
            f"TimesFM-3 prediction failed. Check that covariates shape is (C, T+H) and univariate=False. Error: {exc}"
        ) from exc
    
    # Create result DataFrame
    result = pd.DataFrame({
        'ds': future_df['ds'].iloc[:days].values,
        'yhat': point_forecast,
        'yhat_lower': lower_forecast,
        'yhat_upper': upper_forecast,
    })
    
    # Clip negative values to 0 (traffic can't be negative)
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        result[col] = result[col].clip(lower=0)
    
    return result


def evaluate_timesfm(
    df: pd.DataFrame,
    model_name: str,
    regressors: list[str],
    holdout_days: int,
    checkpoint_path: str = "google/timesfm-3.0-pytorch",
    low_confidence_regressors: set[str] | None = None,
) -> EvaluationResult:
    """
    Evaluate TimesFM model on holdout set.
    """
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
    
    try:
        train_df = df.iloc[:-clamped_holdout].copy()
        valid_df = df.iloc[-clamped_holdout:].copy()
        
        # Fit TimesFM
        model, _ = fit_timesfm(
            train_df,
            regressors,
            checkpoint_path=checkpoint_path,
            low_confidence_regressors=low_confidence_regressors,
        )
        
        # Predict on validation set
        pred_df = predict_timesfm(model, valid_df, days=clamped_holdout)
        
        # Merge predictions with actuals
        merged = valid_df[['ds', 'y']].merge(pred_df[['ds', 'yhat']], on='ds', how='left')
        
        # Calculate metrics
        abs_err = (merged['y'] - merged['yhat']).abs()
        mae = float(abs_err.mean())
        mape = float((abs_err / merged['y'].clip(lower=1)).mean() * 100)
        
        return EvaluationResult(
            model_name=model_name,
            regressors=regressors,
            mae=mae,
            mape=mape,
            holdout_days=int(clamped_holdout),
            status="evaluated",
        )
    
    except Exception as exc:
        return EvaluationResult(
            model_name=model_name,
            regressors=regressors,
            mae=None,
            mape=None,
            holdout_days=0,
            status=f"evaluation_failed: {str(exc)[:100]}",
        )


def train_timesfm_model(
    df: pd.DataFrame,
    holdout_days: int = 14,
    checkpoint_path: str = "google/timesfm-3.0-pytorch",
    low_confidence_regressors: set[str] | None = None,
) -> tuple[object, str, list[str], list[EvaluationResult]]:
    """
    Train TimesFM model with weather + holiday regressors.
    
    Args:
        df: Training DataFrame with 'ds', 'y', and weather regressors
        holdout_days: Days to hold out for validation
        checkpoint_path: TimesFM model checkpoint
        low_confidence_regressors: (not used; kept for API compatibility)
    
    Returns:
        (model, model_name, covariate_names, evaluation_results)
    """
    selected_model_name = "timesfm_weather_holiday"
    selected_regressors = MULTI_REGRESSORS
    
    # Evaluate on holdout if dataset is large enough
    evaluation = evaluate_timesfm(
        df=df,
        model_name=selected_model_name,
        regressors=selected_regressors,
        holdout_days=holdout_days,
        checkpoint_path=checkpoint_path,
        low_confidence_regressors=low_confidence_regressors,
    )
    evaluation_results = [evaluation]
    
    # Train final model on full dataset
    model, covariate_names = fit_timesfm(
        df,
        selected_regressors,
        checkpoint_path=checkpoint_path,
        low_confidence_regressors=low_confidence_regressors,
    )
    
    return model, selected_model_name, covariate_names, evaluation_results


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
