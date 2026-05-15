"""
Unit tests for ModelPredictor.

No checkpoint file is required — all tests use randomly-initialised weights.
EfficientNet-B0 pretrained weights are downloaded by torchvision on first run
(same behaviour as test_smoke.py).
"""

import torch
from PIL import Image

from model import MultiEfficientNet
from src.api.predictor import MAX_CLASS, ModelPredictor

# ── Helpers ───────────────────────────────────────────────────────────


def _make_predictor(num_classes_d: int = 5, num_classes_o: int = 4) -> ModelPredictor:
    """ModelPredictor with random (untrained) weights — no .pth needed."""
    model = MultiEfficientNet(num_classes_d, num_classes_o)
    model.eval()
    idx_to_class_d = {i: i for i in range(num_classes_d)}
    idx_to_class_o = {i: i for i in range(num_classes_o)}
    return ModelPredictor(model, idx_to_class_d, idx_to_class_o, torch.device("cpu"))


def _rgb_image(w: int = 224, h: int = 224) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


# ── predict_batch — output shape ─────────────────────────────────────


def test_predict_batch_single_image_returns_one_result():
    p = _make_predictor()
    results = p.predict_batch([_rgb_image()])
    assert len(results) == 1


def test_predict_batch_returns_one_result_per_image():
    p = _make_predictor()
    results = p.predict_batch([_rgb_image(), _rgb_image()])
    assert len(results) == 2


def test_predict_batch_large_batch():
    p = _make_predictor()
    results = p.predict_batch([_rgb_image() for _ in range(8)])
    assert len(results) == 8


# ── predict_batch — result schema ────────────────────────────────────

EXPECTED_KEYS = {"d_total", "o_occupied", "v_vacant", "confidence_d", "confidence_o", "latency_ms"}


def test_predict_batch_result_has_all_required_keys():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert set(result.keys()) == EXPECTED_KEYS


def test_predict_batch_no_extra_keys():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert set(result.keys()) <= EXPECTED_KEYS


# ── predict_batch — value constraints ────────────────────────────────


def test_confidence_d_is_between_0_and_1():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert 0.0 <= result["confidence_d"] <= 1.0


def test_confidence_o_is_between_0_and_1():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert 0.0 <= result["confidence_o"] <= 1.0


def test_v_vacant_is_non_negative():
    p = _make_predictor()
    for _ in range(10):
        result = p.predict_batch([_rgb_image()])[0]
        assert int(result["v_vacant"]) >= 0


def test_latency_ms_is_positive():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert result["latency_ms"] > 0


# ── predict_batch — label format ─────────────────────────────────────


def test_d_total_label_starts_with_d():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert result["d_total"].startswith("d")


def test_o_occupied_label_starts_with_o():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert result["o_occupied"].startswith("o")


def test_class_boundary_d_7plus():
    """When idx_to_class_d maps to MAX_CLASS (7), label must be 'd7+'."""
    model = MultiEfficientNet(1, 1)
    model.eval()
    p = ModelPredictor(model, {0: MAX_CLASS}, {0: 0}, torch.device("cpu"))
    result = p.predict_batch([_rgb_image()])[0]
    assert result["d_total"] == "d7+"


def test_class_boundary_o_7plus():
    """When idx_to_class_o maps to MAX_CLASS (7), label must be 'o7+'."""
    model = MultiEfficientNet(1, 1)
    model.eval()
    p = ModelPredictor(model, {0: 0}, {0: MAX_CLASS}, torch.device("cpu"))
    result = p.predict_batch([_rgb_image()])[0]
    assert result["o_occupied"] == "o7+"


# ── predict_batch — different input sizes ────────────────────────────


def test_predict_batch_non_square_image_is_resized_correctly():
    """Predictor must handle non-square images (transform resizes to 224×224)."""
    p = _make_predictor()
    result = p.predict_batch([_rgb_image(w=640, h=480)])
    assert len(result) == 1


def test_predict_batch_small_image():
    p = _make_predictor()
    result = p.predict_batch([_rgb_image(w=32, h=32)])
    assert len(result) == 1


# ── ModelPredictor — no NaN in outputs ───────────────────────────────


def test_no_nan_in_confidences():
    import math

    p = _make_predictor()
    result = p.predict_batch([_rgb_image()])[0]
    assert not math.isnan(result["confidence_d"])
    assert not math.isnan(result["confidence_o"])
