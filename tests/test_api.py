import numpy as np
import pytest
from fastapi.testclient import TestClient

import api

VALID = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "No", "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
    "TechSupport": "No internet service", "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 358.2,
}

BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
               "MultipleLines", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "gender"]


class FakeModel:
    """Mimics the sklearn pipeline's output shapes; records what it was given."""
    def __init__(self, pred=1, proba=0.84):
        self.pred, self.proba, self.received = pred, proba, None

    def predict(self, df):
        self.received = df
        return np.array([self.pred])

    def predict_proba(self, df):
        return np.array([[1 - self.proba, self.proba]])


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    model.loader_calls = 0

    def loader():
        model.loader_calls += 1
        return model

    monkeypatch.setattr(api, "load_champion_model", loader)
    return model


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_valid_predict_response_contract(client, fake_model):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_prediction"] == 1
    assert isinstance(body["churn_prediction"], int)
    assert body["churn_probability"] == pytest.approx(0.84)


def test_model_receives_encoded_features(client, fake_model):
    # VALID deliberately uses "No internet service" / "No phone service" so
    # this fails if api.py stops calling the two preprocessing functions.
    client.post("/predict", json=VALID)
    df = fake_model.received
    assert df is not None
    for col in BINARY_COLS:
        assert not df[col].isna().any(), col
        assert set(df[col].unique()) <= {0, 1}, col
    assert not (df == "No internet service").any().any()
    assert not (df == "No phone service").any().any()


@pytest.mark.parametrize("change", [
    {"InternetService": "Satellite"},
    {"tenure": -1},
    {"SeniorCitizen": 2},
    {"MonthlyCharges": -5.0},
    {"gender": None},          # sentinel: handled below as "remove the field"
])
def test_invalid_input_returns_422_and_skips_model(client, fake_model, change):
    payload = dict(VALID)
    for key, value in change.items():
        if value is None:
            del payload[key]
        else:
            payload[key] = value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert fake_model.loader_calls == 0


def test_missing_tracking_uri_returns_503(client, monkeypatch):
    # No mock of the loader: exercises the real error handling.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    r = client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "Model service unavailable" in r.json()["detail"]


def test_model_load_failure_returns_503(client, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid")
    monkeypatch.setattr(api.mlflow, "set_tracking_uri", lambda uri: None)

    def boom(uri):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(api.mlflow.sklearn, "load_model", boom)
    r = client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "registry unreachable" in r.json()["detail"]