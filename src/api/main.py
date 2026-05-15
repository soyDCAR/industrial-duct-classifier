import asyncio
import io
import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog
import structlog.contextvars
import structlog.processors
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import metrics_registry as m
from src.api.predictor import DynamicBatchQueue, ModelPredictor
from src.api.schemas import BatchPredictResponse, HealthResponse, PredictionResult

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_ductos_multitarea_efnet.pth")
MAPPING_PATH = os.getenv("MAPPING_PATH", "class_mapping.json")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

predictor: ModelPredictor | None = None
batch_queue: DynamicBatchQueue | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, batch_queue
    predictor = ModelPredictor.load(MODEL_PATH, MAPPING_PATH)
    batch_queue = DynamicBatchQueue(predictor)
    batch_queue.start()
    yield
    await batch_queue.stop()


app = FastAPI(
    title="Duct Classifier API",
    description="Production-grade ML inference service for industrial duct counting.",
    version="2.0.0",
    lifespan=lifespan,
)


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds maximum size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    _validate_upload(file, data)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot decode image. Ensure the file is a valid image.",
        )


@app.post("/predict", response_model=PredictionResult, summary="Single-image prediction")
async def predict(file: UploadFile = File(...)):
    if batch_queue is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    img = await _read_image(file)

    try:
        result = await batch_queue.predict(img, request_id)
    except Exception as exc:
        logger.error("prediction_failed", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Inference error.", "correlation_id": request_id},
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    result["latency_ms"] = round(latency_ms, 2)
    result["request_id"] = request_id

    m.prediction_latency_seconds.observe(latency_ms / 1000)
    m.predictions_total.labels(task="d", predicted_class=result["d_total"]).inc()
    m.predictions_total.labels(task="o", predicted_class=result["o_occupied"]).inc()
    m.model_confidence.labels(task="d").observe(result["confidence_d"])
    m.model_confidence.labels(task="o").observe(result["confidence_o"])

    logger.info(
        "prediction",
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
        d_total=result["d_total"],
        o_occupied=result["o_occupied"],
        confidence_d=result["confidence_d"],
        confidence_o=result["confidence_o"],
    )

    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch prediction (up to 32 images)",
)
async def predict_batch(files: list[UploadFile] = File(...)):
    if batch_queue is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per request.")

    t0 = time.perf_counter()
    request_ids = [str(uuid.uuid4()) for _ in files]

    images = [await _read_image(f) for f in files]

    raw_results = await asyncio.gather(
        *[batch_queue.predict(img, req_id) for img, req_id in zip(images, request_ids)],
        return_exceptions=True,
    )

    results: list[PredictionResult] = []
    for raw, req_id in zip(raw_results, request_ids):
        if isinstance(raw, Exception):
            raise HTTPException(
                status_code=500,
                detail={"message": "Inference error.", "correlation_id": req_id},
            )
        raw["request_id"] = req_id
        results.append(PredictionResult(**raw))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("batch_prediction", count=len(files), total_latency_ms=total_ms)

    return BatchPredictResponse(results=results, total_latency_ms=total_ms)


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    return HealthResponse(
        status="ok" if predictor is not None else "unavailable",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "N/A",
    )


@app.get("/metrics", summary="Prometheus metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
