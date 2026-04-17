# src/api/metrics.py
# All Prometheus metrics — imported by main.py and predict.py

from prometheus_client import Counter, Histogram, Gauge, Summary, REGISTRY


def _counter(name, doc, labels):
    try:    return Counter(name, doc, labels)
    except: return REGISTRY._names_to_collectors.get(name)

def _histogram(name, doc, labels, buckets):
    try:    return Histogram(name, doc, labels, buckets=buckets)
    except: return REGISTRY._names_to_collectors.get(name)

def _gauge(name, doc, labels=None):
    try:    return Gauge(name, doc, labels or [])
    except: return REGISTRY._names_to_collectors.get(name)

def _summary(name, doc, labels):
    try:    return Summary(name, doc, labels)
    except: return REGISTRY._names_to_collectors.get(name)


# ── application ────────────────────────────────────────────────────────────────
REQUEST_COUNT        = _counter("api_requests_total",          "Total API requests",          ["endpoint","method","status_code"])
ERRORS_TOTAL         = _counter("errors_total",                "Total errors by type",         ["error_type","endpoint"])
IMAGES_PROCESSED     = _counter("images_processed_total",      "Total images processed",       ["status"])
HIGH_RISK_COUNTER    = _counter("high_risk_predictions_total", "High risk predictions",        [])
PREDICTION_CLASS     = _counter("predictions_total",           "Predictions by class",         ["predicted_class","risk_level"])
FEEDBACK_COUNTER     = _counter("feedback_total",              "User feedback votes",          ["vote"])

# ── user level ─────────────────────────────────────────────────────────────────
USER_PREDICTIONS     = _counter("user_predictions_total",      "Predictions per user",         ["username"])
USER_RATE_LIMIT_BREACH = _counter("user_rate_limit_breach_total","Users exceeding 50 req/hr", ["uid","username"])
USERS_REGISTERED     = _gauge("users_registered_total",        "Total registered users")
USERS_VERIFIED       = _gauge("users_verified_total",          "Total verified users")

# ── inference ──────────────────────────────────────────────────────────────────
INFERENCE_LATENCY    = _histogram("inference_latency_seconds", "Inference latency",            ["mode"],          [0.05,0.1,0.25,0.5,1.0,2.0,5.0])
IMG_PROC_DURATION    = _histogram("image_processing_duration_seconds","Preprocessing time",    ["stage"],         [0.001,0.005,0.01,0.05,0.1,0.5])
CONFIDENCE_HIST      = _histogram("prediction_confidence",     "Confidence distribution",      ["predicted_class"],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
REQ_MEMORY           = _histogram("request_memory_usage_mb",   "Memory per request (MB)",      ["endpoint"],      [50,100,200,500,1000,2000])
REQ_CPU              = _histogram("request_cpu_usage_percent", "CPU per request",              ["endpoint"],      [5,10,20,30,50,70,90])
INFER_SUMMARY        = _summary("inference_latency_summary",   "Inference latency summary",    ["mode"])

# ── system ─────────────────────────────────────────────────────────────────────
MODEL_LOADED         = _gauge("model_loaded",                  "1 if model ready")
ACTIVE_REQUESTS      = _gauge("active_requests",               "Active requests",              ["endpoint"])
CPU_PERCENT          = _gauge("system_cpu_percent",            "System CPU %")
MEMORY_PERCENT       = _gauge("system_memory_percent",         "System memory %")
MEMORY_USED_GB       = _gauge("system_memory_used_gb",         "System memory used (GB)")
GPU_ALLOC            = _gauge("gpu_memory_allocated_gb",       "GPU memory allocated (GB)")
GPU_RESERVED         = _gauge("gpu_memory_reserved_gb",        "GPU memory reserved (GB)")
MODEL_MEMORY         = _gauge("model_memory_bytes",            "Model parameter bytes")
UPTIME               = _gauge("api_uptime_seconds",            "API uptime (seconds)")
MONGODB_UP           = _gauge("mongodb_up",                    "1 if MongoDB reachable")

# ── user feedback ──────────────────────────────────────────────────────────────
FEEDBACK_RATE        = _gauge("feedback_positive_rate",        "Rolling positive feedback rate")
UNIQUE_PREDS         = _gauge("unique_predictions_total",      "Total unique predictions")