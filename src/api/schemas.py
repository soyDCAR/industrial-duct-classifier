from pydantic import BaseModel


class PredictionResult(BaseModel):
    d_total: str
    o_occupied: str
    v_vacant: str
    confidence_d: float
    confidence_o: float
    latency_ms: float
    request_id: str


class BatchPredictResponse(BaseModel):
    results: list[PredictionResult]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
