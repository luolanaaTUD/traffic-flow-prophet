from __future__ import annotations

import pytest

from app.service import TrafficModelService


def _make_test_records(n_days: int = 60):
    """Helper to create varied test records that pass quality gates."""
    import pandas as pd
    base_date = pd.Timestamp("2026-01-01")
    
    records = []
    for i in range(n_days):
        records.append({
            "ds": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "y": 1000 + (i % 10) * 100,
            "temp_max": 20.0 + (i % 15),
            "temp_min": 15.0 + (i % 10),
            "precip": float(i % 5),
            "humidity": 50.0 + (i % 30),
            "pressure": 1010.0 + (i % 20),
            "vis": 15.0 + (i % 15),
            "cloud": 30.0 + (i % 50),
            "uv_index": float(i % 11),
            "wind_speed_day": 5.0 + (i % 20),
            "wind_speed_night": 4.0 + (i % 15),
        })
    return records


def test_default_model_is_prophet():
    """
    Test that the default model is Prophet (production-safe).
    TimesFM is now opt-in due to non-commercial license.
    """
    service = TrafficModelService()
    records = _make_test_records()
    
    # Train with default (should be Prophet) - no model_name specified
    result = service.train_from_records(
        records=records,
        holdout_days=14,
        max_training_days=60,
        # NO model_name - should default to multi_weather_regressors
    )
    
    # Verify default is Prophet
    assert result["model_name"] == "multi_weather_regressors"
    assert result["rows"] == 60


def test_timesfm_explicit_selection_without_package():
    """
    Test that explicitly selecting TimesFM model without timesfm package installed
    gives a helpful error message (not a generic import error).
    """
    service = TrafficModelService()
    records = _make_test_records()
    
    # Try to train with TimesFM explicitly - should fail gracefully if not installed
    try:
        result = service.train_from_records(
            records=records,
            holdout_days=14,
            max_training_days=60,
            model_name="timesfm_weather_holiday",
        )
        # If we get here, timesfm is installed - that's fine too
        assert result["model_name"] == "timesfm_weather_holiday"
    except ImportError as exc:
        # Expected when timesfm is not installed
        assert "timesfm" in str(exc).lower()
        assert "pip install" in str(exc).lower()
    except ValueError as exc:
        # Could be quality gate or other validation
        error_msg = str(exc).lower()
        assert any(keyword in error_msg for keyword in ["variance", "quality", "timesfm", "checkpoint"])


def test_prophet_opt_in_works_without_timesfm():
    """
    Test that explicitly selecting Prophet works even without timesfm installed.
    This is the production-safe opt-in path.
    """
    service = TrafficModelService()
    records = _make_test_records()
    
    # Explicitly use Prophet - should always work
    result = service.train_from_records(
        records=records,
        holdout_days=14,
        max_training_days=60,
        model_name="multi_weather_regressors",
    )
    
    assert result["model_name"] == "multi_weather_regressors"
    assert result["rows"] == 60


def test_invalid_model_name_raises_value_error():
    """Test that invalid model names are rejected."""
    service = TrafficModelService()
    records = _make_test_records()
    
    with pytest.raises(ValueError, match="Invalid model_name"):
        service.train_from_records(
            records=records,
            holdout_days=14,
            max_training_days=60,
            model_name="invalid_model_name",
        )


def test_prophet_no_zero_predictions_on_positive_history():
    """
    Regression test: Prophet with log1p/expm1 transform should never predict
    zero tourists when trained on positive historical data.
    
    This tests the fix for the issue where negative Prophet yhat values
    were clipped to 0, incorrectly predicting zero tourists.
    """
    import pandas as pd
    
    service = TrafficModelService()
    
    # Create synthetic data with positive tourist counts
    # Include some variability that might cause additive Prophet to go negative
    records = []
    base_date = pd.Timestamp("2026-01-01")
    for i in range(90):
        # Simulate varying tourist counts (always positive)
        y_value = 2000 + (i % 30) * 100 - (i % 7) * 50
        records.append({
            "ds": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "y": max(1500, y_value),  # Ensure all positive
            "temp_max": 22.0 + (i % 12) - 3,
            "temp_min": 16.0 + (i % 8) - 2,
            "precip": float((i % 5) * 0.5),
            "humidity": 55.0 + (i % 25),
            "pressure": 1012.0 + (i % 18) - 4,
            "vis": 18.0 + (i % 12),
            "cloud": 35.0 + (i % 45),
            "uv_index": float((i % 10)),
            "wind_speed_day": 8.0 + (i % 15),
            "wind_speed_night": 6.0 + (i % 12),
        })
    
    # Train Prophet model
    result = service.train_from_records(
        records=records,
        holdout_days=14,
        max_training_days=90,
        model_name="multi_weather_regressors",
    )
    
    assert result["model_name"] == "multi_weather_regressors"
    
    # Create mock future weather data
    future_records = []
    for i in range(7):
        future_records.append({
            "fxDate": (base_date + pd.Timedelta(days=90 + i)).strftime("%Y-%m-%d"),
            "tempMax": "20",
            "tempMin": "15",
            "precip": "0.5",
            "humidity": "60",
            "pressure": "1015",
            "vis": "20",
            "cloud": "40",
            "uvIndex": "5",
            "windSpeedDay": "10",
            "windSpeedNight": "8",
        })
    
    # Predict next 7 days
    predictions = service.predict_next_days(future_records, days=7)
    
    # Verify no zero predictions
    assert len(predictions["predictions"]) == 7
    for pred in predictions["predictions"]:
        # With log transform, predictions should always be positive
        assert pred["yhat"] > 0, f"Predicted zero tourists on {pred['ds']}"
        assert pred["yhat_lower"] >= 0, f"Lower bound negative on {pred['ds']}"
        assert pred["yhat_upper"] > 0, f"Upper bound zero on {pred['ds']}"
        
        # Sanity check: predictions should be reasonable given training data
        assert pred["yhat"] > 500, f"Suspiciously low prediction on {pred['ds']}: {pred['yhat']}"
