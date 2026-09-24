"""API-level tests for the Task 10 FastAPI service.

Ran against the FastAPI TestClient, so no container is required:
    pytest tests/
"""

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

VALID = {
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0.0,
    "product_type": "L",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["loaded"] is True


def test_valid_input_returns_200_with_expected_shape():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "predicted_failure_type",
        "confidence",
        "risk",
        "decision_threshold",
        "failure_probabilities",
    }

    assert isinstance(body["predicted_failure_type"], str)
    assert body["predicted_failure_type"] in main.CLASS_NAMES

    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["risk"], float)
    assert 0.0 <= body["risk"] <= 1.0

    assert body["decision_threshold"] == pytest.approx(
        main.DECISION_THRESHOLD, rel=1e-2
    )

    probs = body["failure_probabilities"]
    assert isinstance(probs, dict)
    assert list(probs) == main.CLASS_NAMES
    assert len(probs) == 6
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-2)


def test_invalid_type_returns_400():
    payload = dict(VALID)
    payload["torque"] = "not-a-number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_missing_required_field_returns_400():
    payload = dict(VALID)
    del payload["tool_wear"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_out_of_range_value_returns_400():
    payload = dict(VALID)
    payload["air_temperature"] = 50.0  # far below the physical sensor envelope
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Request payload failed validation."
    assert any("air_temperature" in str(err["loc"]) for err in body["errors"])


def test_physical_constraint_process_below_air_returns_400():
    payload = dict(VALID)
    payload["air_temperature"] = 320.0
    payload["process_temperature"] = 300.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 400