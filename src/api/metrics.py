# src/api/metrics.py
# All Prometheus metrics — single source of truth
# Imported by main.py and predict.py

from prometheus_client import Counter, Histogram, Gauge, Summary, REGISTRY


def _counter(name, doc, labels):
    try:
        return Counter(name, doc, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _histogram(name, doc, labels, buckets):
    try:
        return Histogram(name, doc, labels, buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _gauge(name, doc, labels=None):
    try:
        return Gauge(name, doc, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _summary(name, doc, labels):
    try:
        return Summary(name, doc, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


# ── Application level ──────────────────────────────────────────────────────────

# every HTTP request — labels allow breakdown by endpoint, method, status
REQUEST_COUNT = _counter(
    "api_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status_code"]
)

# error tracking by type and endpoint
ERRORS_TOTAL = _counter(
    "errors_total",
    "Total errors by type",
    ["error_type", "endpoint"]
)

# image processing results
IMAGES_PROCESSED = _counter(
    "images_processed_total",
    "Total images processed",
    ["status"]   # success | failed
)

# FIX: HIGH_RISK_COUNTER had empty labels [] causing inc() to fail silently
# Use labels=[] and call .inc() with no arguments — this is correct
HIGH_RISK_COUNTER = _counter(
    "high_risk_predictions_total",
    "Predictions classified as HIGH risk (melanoma)",
    []
)

# per class prediction counter — use sum(predictions_total) in Prometheus
PREDICTION_CLASS = _counter(
    "predictions_total",
    "Predictions by disease class",
    ["predicted_class", "risk_level"]
)

# user feedback votes
FEEDBACK_COUNTER = _counter(
    "feedback_total",
    "User feedback votes",
    ["vote"]   # thumbs_up | thumbs_down
)


# ── User level ─────────────────────────────────────────────────────────────────

# per-user prediction counter — use sum(user_predictions_total) in Prometheus
USER_PREDICTIONS = _counter(
    "user_predictions_total",
    "Predictions per user",
    ["username"]
)

# rate limit breach counter — fires when user exceeds request threshold
USER_RATE_LIMIT_BREACH = _counter(
    "user_rate_limit_breach_total",
    "Users exceeding request rate limit",
    ["uid", "username"]
)

# total users gauge — updated by background thread in main.py
USERS_REGISTERED = _gauge(
    "users_registered_total",
    "Total registered users"
)

USERS_VERIFIED = _gauge(
    "users_verified_total",
    "Total email-verified users"
)


# ── Inference level ────────────────────────────────────────────────────────────

# model inference time — use histogram_quantile(0.95, ...) in Prometheus
INFERENCE_LATENCY = _histogram(
    "inference_latency_seconds",
    "End-to-end model inference latency",
    ["mode"],
    [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# image preprocessing time broken into stages
IMG_PROC_DURATION = _histogram(
    "image_processing_duration_seconds",
    "Image preprocessing time by stage",
    ["stage"],   # decode_resize | normalize_tensor
    [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# model output confidence scores — use histogram_quantile(0.50, ...) in Prometheus
CONFIDENCE_HIST = _histogram(
    "prediction_confidence",
    "Model prediction confidence score distribution",
    ["predicted_class"],
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# memory consumed per request
REQ_MEMORY = _histogram(
    "request_memory_usage_mb",
    "Memory delta per request (MB)",
    ["endpoint"],
    [10, 50, 100, 200, 500, 1000, 2000]
)

# CPU consumed per request
REQ_CPU = _histogram(
    "request_cpu_usage_percent",
    "CPU usage snapshot per request",
    ["endpoint"],
    [5, 10, 20, 30, 50, 70, 90]
)

# latency summary for p50/p95/p99 in a single metric
INFER_SUMMARY = _summary(
    "inference_latency_summary",
    "Summary of inference latency (p50, p95, p99)",
    ["mode"]
)


# ── System level — updated by background thread every 5s ──────────────────────

MODEL_LOADED = _gauge(
    "model_loaded",
    "1 if ML model is loaded and ready, 0 if not"
)

# per-endpoint concurrent request gauge
ACTIVE_REQUESTS = _gauge(
    "active_requests",
    "Currently active (in-flight) requests",
    ["endpoint"]
)

# system CPU — set by background thread, not by middleware
# FIX: was set in middleware only on requests → missing data between requests
CPU_PERCENT = _gauge(
    "system_cpu_percent",
    "System CPU utilization percent (updated every 5s)"
)

# system memory
MEMORY_PERCENT = _gauge(
    "system_memory_percent",
    "System memory utilization percent (updated every 5s)"
)

MEMORY_USED_GB = _gauge(
    "system_memory_used_gb",
    "System memory used in gigabytes (updated every 5s)"
)

# GPU metrics (only meaningful on GPU machines — stays at 0 on CPU)
GPU_ALLOC = _gauge(
    "gpu_memory_allocated_gb",
    "GPU memory currently allocated (GB)"
)

GPU_RESERVED = _gauge(
    "gpu_memory_reserved_gb",
    "GPU memory reserved by PyTorch cache (GB)"
)

# model size in bytes (set once at startup)
MODEL_MEMORY = _gauge(
    "model_memory_bytes",
    "ML model parameter memory footprint in bytes"
)

# API uptime — updated every 5s by background thread
UPTIME = _gauge(
    "api_uptime_seconds",
    "API server uptime in seconds"
)

# MongoDB connectivity — 1=up, 0=down (updated every 5s)
MONGODB_UP = _gauge(
    "mongodb_up",
    "1 if MongoDB is reachable, 0 if not"
)


# ── User feedback level ────────────────────────────────────────────────────────

# rolling positive feedback ratio gauge (0.0 to 1.0)
FEEDBACK_RATE = _gauge(
    "feedback_positive_rate",
    "Rolling ratio of thumbs_up / total feedback"
)

# total unique predictions counter (monotonically increasing gauge)
UNIQUE_PREDS = _gauge(
    "unique_predictions_total",
    "Total unique prediction requests served"
)