"""
Integration tests for the FastAPI inference service.
The model is mocked so no .pth file is needed.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

MOCK_RESULT = {
    "d_total": "d2",
    "o_occupied": "o1",
    "v_vacant": "1",
    "confidence_d": 0.85,
    "confidence_o": 0.90,
    "latency_ms": 12.5,
}


def _jpeg_bytes(size: int = 224) -> bytes:
    img = Image.new("RGB", (size, size), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client():
    mock_predictor = MagicMock()
    mock_predictor.device = torch.device("cpu")

    mock_queue = MagicMock()
    mock_queue.predict = AsyncMock(return_value=MOCK_RESULT.copy())
    mock_queue.stop = AsyncMock()
    mock_queue.start = MagicMock()

    with (
        patch("src.api.main.ModelPredictor") as MockPredictor,
        patch("src.api.main.DynamicBatchQueue") as MockQueue,
    ):
        MockPredictor.load.return_value = mock_predictor
        MockQueue.return_value = mock_queue

        with TestClient(app) as c:
            yield c


# ── /health ──────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


# ── /predict ─────────────────────────────────────────────────────────


def test_predict_valid_jpeg(client):
    r = client.post("/predict", files={"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["d_total"] == "d2"
    assert body["o_occupied"] == "o1"
    assert "request_id" in body
    assert isinstance(body["latency_ms"], float)


def test_predict_valid_png(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    r = client.post("/predict", files={"file": ("img.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200


def test_predict_unsupported_mime_returns_415(client):
    r = client.post("/predict", files={"file": ("doc.pdf", b"fake-pdf", "application/pdf")})
    assert r.status_code == 415


def test_predict_too_large_returns_413(client):
    big = b"x" * (11 * 1024 * 1024)
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_predict_corrupt_image_returns_400(client):
    r = client.post("/predict", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400


# ── /predict/batch ────────────────────────────────────────────────────


def test_predict_batch_two_images(client):
    img = _jpeg_bytes()
    files = [
        ("files", ("a.jpg", img, "image/jpeg")),
        ("files", ("b.jpg", img, "image/jpeg")),
    ]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert "total_latency_ms" in body
    for result in body["results"]:
        assert "request_id" in result


def test_predict_batch_over_limit_returns_400(client):
    img = _jpeg_bytes()
    files = [("files", (f"{i}.jpg", img, "image/jpeg")) for i in range(33)]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 400


# ── /metrics ─────────────────────────────────────────────────────────


def test_metrics_endpoint_returns_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text
    assert "prediction_latency_seconds" in r.text
