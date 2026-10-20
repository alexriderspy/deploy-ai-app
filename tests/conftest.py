import time

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import settings
from app.main import create_app


class FakeClassifier:
    """Stands in for Llama in tests: instant, deterministic, and it records every batch it gets."""
    device = "cpu"

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.batches: list[int] = []

    def predict(self, issues: list[tuple[str, str]]) -> list[tuple[str, float]]:
        time.sleep(self.delay_s)
        self.batches.append(len(issues))
        return [("bug" if "crash" in title.lower() else "usage", 0.9) for title, _ in issues]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", SecretStr("test-key"))
    with TestClient(create_app(load=FakeClassifier)) as c:     # "with" runs the lifespan
        yield c
