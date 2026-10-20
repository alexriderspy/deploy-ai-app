from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import settings
from app.main import create_app
from tests.conftest import FakeClassifier

KEY = {"X-API-Key": "test-key"}
ISSUE = {"title": "Server crashes on startup with Qwen3", "body": "Traceback ..."}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_label(client):
    r = client.post("/predict", json=ISSUE, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "bug"
    assert 0 <= body["confidence"] <= 1
    assert r.headers["x-request-id"]


def test_missing_or_wrong_key_is_rejected(client):
    assert client.post("/predict", json=ISSUE).status_code == 401
    assert client.post("/predict", json=ISSUE, headers={"X-API-Key": "nope"}).status_code == 401


def test_bad_input_is_a_422(client):
    assert client.post("/predict", json={"title": ""}, headers=KEY).status_code == 422
    assert client.post("/predict", json={"body": "no title"}, headers=KEY).status_code == 422


def test_metrics_count_requests(client):
    client.post("/predict", json=ISSUE, headers=KEY)
    text = client.get("/metrics").text
    assert 'triage_requests_total{status="ok"}' in text
    assert 'triage_predictions_total{label="bug"}' in text


def test_slow_model_times_out(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", SecretStr(""))
    monkeypatch.setattr(settings, "request_timeout_s", 0.1)
    with TestClient(create_app(load=lambda: FakeClassifier(delay_s=0.5))) as c:
        assert c.post("/predict", json=ISSUE).status_code == 504
