from prometheus_client import Counter, Histogram

predictions_total = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["task", "predicted_class"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

model_confidence = Histogram(
    "model_confidence",
    "Model softmax confidence for the predicted class",
    ["task"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
