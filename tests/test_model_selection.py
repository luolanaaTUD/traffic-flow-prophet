from __future__ import annotations

import pytest

from app.service import TrafficModelService


def test_timesfm_model_selection_without_timesfm_installed():
    """
    Test that selecting TimesFM model without timesfm package installed
    gives a helpful error message (not a generic import error).
    """
    service = TrafficModelService()
    
    # Create varied training data (avoid low-variance quality gates)
    import pandas as pd
    base_date = pd.Timestamp("2026-01-01")
    
    records = []
    for i in range(60):
        records.append({
            "ds": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "y": 1000 + (i % 10) * 100,  # Varied traffic
            "temp_max": 20.0 + (i % 15),  # Varied temp
            "temp_min": 15.0 + (i % 10),
            "precip": float(i % 5),  # Varied precipitation
            "humidity": 50.0 + (i % 30),  # Varied humidity
            "pressure": 1010.0 + (i % 20),  # Varied pressure
            "vis": 15.0 + (i % 15),  # Varied visibility
            "cloud": 30.0 + (i % 50),  # Varied cloud cover
            "uv_index": float(i % 11),  # Varied UV
            "wind_speed_day": 5.0 + (i % 20),  # Varied wind
            "wind_speed_night": 4.0 + (i % 15),
        })
    
    # Try to train with TimesFM - should fail gracefully if not installed
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
        # Allow quality gates or missing timesfm
        assert any(keyword in error_msg for keyword in ["variance", "quality", "timesfm", "checkpoint"])


def test_invalid_model_name_raises_value_error():
    """Test that invalid model names are rejected."""
    service = TrafficModelService()
    
    import pandas as pd
    base_date = pd.Timestamp("2026-01-01")
    
    records = []
    for i in range(60):
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
    
    with pytest.raises(ValueError, match="Invalid model_name"):
        service.train_from_records(
            records=records,
            holdout_days=14,
            max_training_days=60,
            model_name="invalid_model_name",
        )
