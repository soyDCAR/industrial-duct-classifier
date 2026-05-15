# syntax=docker/dockerfile:1
#
# ── Duct Classifier — Multi-Stage Optimised Build ─────────────────────
#
# Stage sizes (approximate, uncompressed):
#   builder-api  — not shipped — ~1.3 GB  (pip cache + build tools)
#   builder-demo — not shipped — ~2.0 GB  (+ gradio / matplotlib)
#   api          — FINAL       — ~800 MB  ← production target
#   demo         — FINAL       — ~1.5 GB  ← Gradio demo target
#
# Why the jump from ~4 GB → ~800 MB for the API image:
#   Removed transformers  (~500 MB)  not used by inference
#   Removed gradio        (~300 MB)  demo only → separate stage
#   Removed torchaudio    (~100 MB)  not used at all
#   Removed scikit-learn  (~100 MB)  training only
#   Removed scipy         (~100 MB)  training only
#   Removed pandas        ( ~80 MB)  training only
#   Removed matplotlib    ( ~50 MB)  training only
#   Removed opencv        (~100 MB)  training/demo only
#   Multi-stage strips pip cache and build tools from the final layer
#
# Build:
#   docker build --target api  -t ductos:api  .
#   docker build --target demo -t ductos:demo .
#
# Run API:
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/modelo_ductos_multitarea_efnet.pth:/app/modelo_ductos_multitarea_efnet.pth:ro \
#     -v $(pwd)/class_mapping.json:/app/class_mapping.json:ro \
#     ductos:api
#
# Run Gradio demo:
#   docker run --rm -p 7860:7860 \
#     -v $(pwd)/modelo_ductos_multitarea_efnet.pth:/app/modelo_ductos_multitarea_efnet.pth:ro \
#     -v $(pwd)/class_mapping.json:/app/class_mapping.json:ro \
#     ductos:demo

# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — builder-api
# Installs torch CPU + API dependencies into an isolated virtual env.
# This stage is never shipped; only /opt/venv is copied forward.
# ═══════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS builder-api

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch CPU wheels first — largest layer; cached separately so reruns skip it
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Minimal API deps — see requirements-api.txt for the full exclusion list
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — builder-demo
# Extends builder-api with Gradio and matplotlib for the interactive demo.
# Also not shipped.
# ═══════════════════════════════════════════════════════════════════════
FROM builder-api AS builder-demo

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — api  (PRODUCTION, ~800 MB)
# Clean python:3.10-slim base + pre-built venv + minimal source.
# No pip, no gcc, no build cache, no training or demo code.
# ═══════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS api

LABEL maintainer="Dilan Acosta"
LABEL description="Production-grade ML inference service: PyTorch model served via FastAPI with batching, monitoring, and load tests."
LABEL version="2.0"

WORKDIR /app

COPY --from=builder-api /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Only the files the API actually imports at runtime
COPY model.py ./
COPY src/ src/

# Model weights and class_mapping.json are mounted as volumes — not baked in
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ═══════════════════════════════════════════════════════════════════════
# Stage 4 — demo  (~1.5 GB)
# Adds Gradio + matplotlib on top of the api base.
# Use for interactive demos only; not suitable for production traffic.
# ═══════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS demo

LABEL maintainer="Dilan Acosta"
LABEL description="Gradio demo — interactive exploration. For production use the 'api' target."
LABEL version="2.0"

WORKDIR /app

COPY --from=builder-demo /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY model.py predict.py ./
COPY src/ src/
COPY demo/ demo/

EXPOSE 7860
CMD ["python", "demo/app.py", "--port", "7860"]
