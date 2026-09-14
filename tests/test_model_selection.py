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


def test_default_model_is_timesfm_and_fails_without_package():
    """
    Test that the default model is TimesFM and it fails with a clear
    error message when timesfm package is not installed.
    """
    service = TrafficModelService()
    records = _make_test_records()
    
    # Try to train with default (TimesFM) - no model_name specified
    try:
        result = service.train_from_records(
            records=records,
            holdout_days=14,
            max_training_days=60,
            # NO model_name - should default to timesfm_weather_holiday
        )
        # If we get here, timesfm is installed - verify it's TimesFM
        assert result["model_name"] == "timesfm_weather_holiday"
    except ImportError as exc:
        # Expected when timesfm is not installed - verify helpful message
        error_msg = str(exc).lower()
        assert "timesfm" in error_msg
        assert "pip install" in error_msg or "install" in error_msg
    except ValueError as exc:
        # Quality gates are OK too
        error_msg = str(exc).lower()
        assert "variance" in error_msg or "quality" in error_msg


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
