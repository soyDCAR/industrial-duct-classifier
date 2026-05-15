"""
Locust load test — Duct Classifier API.

Requirements:
    pip install locust

Run headless (recommended for CI / baseline measurement):
    locust -f locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 10 --run-time 60s --headless

Run with web UI (real-time charts at http://localhost:8089):
    locust -f locustfile.py --host http://localhost:8000

Target SLO:
    Throughput : 200 RPS (with 50 concurrent users on CPU)
    Latency    : p95 < 500 ms  /predict single
                 p95 < 800 ms  /predict/batch (4 images)
"""

import io
import random

from locust import HttpUser, between, task
from PIL import Image

# ── Shared synthetic image ─────────────────────────────────────────────


def _jpeg_bytes(size: int = 224) -> bytes:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── User behaviour ────────────────────────────────────────────────────


class DuctAPIUser(HttpUser):
    """
    Task weights model realistic traffic:
      80 % single predictions  (most clients send one image at a time)
      15 % batch predictions   (pipeline / bulk clients)
       5 % health checks       (load balancer probes)
    """

    wait_time = between(0.05, 0.3)

    def on_start(self) -> None:
        # Generate a fixed image per user so we're measuring inference, not PIL overhead
        self._single_img = _jpeg_bytes()
        self._batch_imgs = [_jpeg_bytes() for _ in range(4)]

    @task(16)
    def predict_single(self) -> None:
        self.client.post(
            "/predict",
            files={"file": ("img.jpg", self._single_img, "image/jpeg")},
            name="/predict",
        )

    @task(3)
    def predict_batch(self) -> None:
        files = [("files", ("img.jpg", img, "image/jpeg")) for img in self._batch_imgs]
        self.client.post("/predict/batch", files=files, name="/predict/batch [4 imgs]")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")
