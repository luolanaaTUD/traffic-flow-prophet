from __future__ import annotations

import pytest


def test_timesfm_import_uses_correct_api():
    """
    Test that TimesFM import uses the TimesFM-3 API, not the old 1.x/2.x API.
    This prevents the bug: "module 'timesfm' has no attribute 'TimesFm'"
    """
    try:
        from app.modeling_timesfm import _try_import_timesfm
        
        # Try to import - should get TimesFM-3 classes
        ModelConfig, TimesFM3Evaluator = _try_import_timesfm()
        
        # Verify we got the right classes
        assert ModelConfig is not None
        assert TimesFM3Evaluator is not None
        
        # Verify class names (shouldn't be the old API)
        assert "ModelConfig" in str(ModelConfig)
        assert "TimesFM3" in str(TimesFM3Evaluator) or "Evaluator" in str(TimesFM3Evaluator)
        
    except ImportError as exc:
        # Expected if timesfm not installed - verify error message
        error_msg = str(exc).lower()
        assert "timesfm" in error_msg
        assert "install" in error_msg
        # Make sure we're NOT trying to use old API
        assert "timesfm" not in error_msg or "timesfm-3" in error_msg or "timesfm3" in error_msg or "timesfm[torch]" in error_msg


def test_timesfm_does_not_use_old_api():
    """
    Verify that modeling_timesfm.py does NOT reference the old TimesFM 1.x/2.x API.
    """
    from pathlib import Path
    
    # Read the modeling_timesfm.py file
    timesfm_file = Path(__file__).parent.parent / "app" / "modeling_timesfm.py"
    content = timesfm_file.read_text()
    
    # Check that old API is NOT used
    assert "timesfm.TimesFm(" not in content, "Old API 'timesfm.TimesFm' found in code"
    assert "load_from_checkpoint" not in content, "Old API 'load_from_checkpoint' found in code"
    assert ".forecast(" not in content or "model.forecast" not in content, "Old API '.forecast' method found"
    
    # Check that new API IS used
    assert "TimesFM3Evaluator" in content or "TimesFM3Forecaster" in content, "TimesFM-3 API not found"
    assert "ModelConfig" in content, "TimesFM-3 ModelConfig not found"
    assert "predict_batch" in content, "TimesFM-3 predict_batch method not found"
    assert "univariate=False" in content, "univariate=False (required for covariates) not found"


def test_timesfm_covariate_shape_documentation():
    """
    Verify that covariate shapes are correctly documented in the code.
    TimesFM-3 expects (C, T+H) for past_future_covariates, not (1, T, C).
    """
    from pathlib import Path
    
    timesfm_file = Path(__file__).parent.parent / "app" / "modeling_timesfm.py"
    content = timesfm_file.read_text()
    
    # Check for correct shape documentation
    assert "(C, T)" in content or "(num_features, time_steps)" in content, "Correct covariate shape (C, T) not documented"
    assert "(C, T+H)" in content or "context_len + horizon_len" in content, "Correct combined covariate shape not documented"
    
    # Make sure we're not using wrong shapes
    assert "(1, T, C)" not in content, "Wrong covariate shape (1, T, C) found"
    assert "(batch, time, features)" not in content, "Wrong covariate shape convention found"
