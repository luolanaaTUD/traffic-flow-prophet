from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_processing import (
    load_training_csv,
    records_to_dataframe,
    validate_training_df,
)
from app.main import app
from app.service import TrafficModelService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = "data/historical_flow_from_summary.csv"


def _csv_records() -> list[dict]:
    raw_df = load_training_csv(DEFAULT_CSV)
    return json.loads(raw_df.to_json(orient="records", date_format="iso"))


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def service() -> TrafficModelService:
    return TrafficModelService()


def test_records_to_dataframe_matches_csv_validation() -> None:
    csv_df = load_training_csv(DEFAULT_CSV)
    records = _csv_records()
    records_df = records_to_dataframe(records)

    csv_validated = validate_training_df(csv_df)
    records_validated = validate_training_df(records_df)

    assert list(records_validated.columns) == list(csv_validated.columns)
    assert len(records_validated) == len(csv_validated)
    pd.testing.assert_frame_equal(
        records_validated.sort_values("ds").reset_index(drop=True),
        csv_validated.sort_values("ds").reset_index(drop=True),
        check_dtype=False,
    )


def test_train_from_records_matches_csv_row_count(service: TrafficModelService) -> None:
    records = _csv_records()
    records_result = service.train_from_records(records=records, max_training_days=60)

    csv_service = TrafficModelService()
    csv_result = csv_service.train_from_csv(csv_path=DEFAULT_CSV, max_training_days=60)

    assert records_result["rows"] == csv_result["rows"]
    assert records_result["model_name"] == csv_result["model_name"]


def test_train_api_with_records(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "records": _csv_records(),
            "holdout_days": 14,
            "max_training_days": 60,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "multi_weather_regressors"
    assert body["rows"] >= 60


def test_train_api_rejects_csv_path_on_train_endpoint(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "records": _csv_records(),
            "csv_path": DEFAULT_CSV,
            "holdout_days": 14,
            "max_training_days": 60,
        },
    )
    assert response.status_code == 422


def test_train_api_empty_records_returns_422(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "records": [],
            "holdout_days": 14,
            "max_training_days": 60,
        },
    )
    assert response.status_code == 422


def test_train_from_csv_api(client: TestClient) -> None:
    response = client.post(
        "/train/from-csv",
        json={
            "csv_path": DEFAULT_CSV,
            "holdout_days": 14,
            "max_training_days": 60,
        },
    )
    assert response.status_code == 200
    assert response.json()["rows"] >= 60


def test_train_from_csv_api_missing_file_returns_404(client: TestClient) -> None:
    response = client.post(
        "/train/from-csv",
        json={
            "csv_path": "data/does-not-exist.csv",
            "holdout_days": 14,
            "max_training_days": 60,
        },
    )
    assert response.status_code == 404
