# src/api/main.py
# FastAPI app entry point — wires auth, predict, metrics routers
# uvicorn src.api.main:app --host 0.0.0.0 --port 8000

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True
)

import os
import time
import platform
import threading
from contextlib import asynccontextmanager

import torch
import psutil
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from torchvision import transforms

from src.models.model import build_model
from src.db.mongodb import mongo
from src.api import metrics as M
from src.api.auth import router as auth_router
from src.api.predict import router as predict_router, set_app_state, CLASS_NAMES, DISEASE_META
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.utils.config import load_params, resolve_path

logger = get_logger("api")

IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE", 224))
PARAMS        = load_params()
MODEL_PATH    = os.getenv("MODEL_PATH", resolve_path(PARAMS["evaluate"]["model_path"]))
MODEL_NAME  = os.getenv("MODEL_NAME", "efficientnet_b3")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 7))
DEVICE      = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "")

app_state = {
    "model"     : None,
    "transform" : None,
    "start_time": None,
    "pred_count": 0,
    "device"    : DEVICE,
    "image_size": IMAGE_SIZE,
    "_stop_bg"  : threading.Event(),   # signal to stop background thread
}

# warmup psutil CPU so first reading is valid
psutil.cpu_percent(interval=None)



def _restore_metrics_from_db():
    """
    Restore persistent counters from MongoDB on startup.
    Without this, Prometheus gauges reset to 0 every time backend restarts,
    even though the actual data is still in MongoDB.

    Affected metrics:
      - unique_predictions_total  → count of all predictions in DB
      - feedback_positive_rate    → computed from feedback collection
      - users_registered_total    → count of users in DB
      - users_verified_total      → count of verified users in DB
    """
    try:
        if not mongo.is_up():
            logger.warning("MongoDB not available — skipping metric restore")
            return

        # predictions
        stats = mongo.get_prediction_stats()
        total_preds = stats.get("total_predictions", 0)
        M.UNIQUE_PREDS.set(total_preds)
        app_state["pred_count"] = total_preds
        logger.info(f"Restored unique_predictions_total = {total_preds}")

        # feedback rate
        fb = mongo.get_feedback_stats()
        if fb.get("total", 0) > 0:
            M.FEEDBACK_RATE.set(fb["positive_rate"])
            logger.info(f"Restored feedback_positive_rate = {fb['positive_rate']:.3f}")

        # user counts
        reg = mongo.get_all_users_count()
        ver = mongo.get_verified_users_count()
        M.USERS_REGISTERED.set(reg)
        M.USERS_VERIFIED.set(ver)
        logger.info(f"Restored users: {reg} registered, {ver} verified")

        # per-class prediction counters from DB aggregation
        # These are Counters so we can only set them via _metrics_total hack
        # Instead, log the breakdown for reference — Prometheus will track
        # increments from this point forward correctly
        if stats.get("by_class"):
            logger.info(f"Class distribution from DB: {stats['by_class']}")

    except Exception as e:
        logger.warning(f"Metric restore from DB failed (non-fatal): {e}")


def _background_metrics_loop(stop_event: threading.Event):
    """
    Background thread — updates system metrics every 5 seconds.
    Runs independently of HTTP traffic so Prometheus always has fresh values.

    WHY this is needed:
      - Prometheus scrapes /metrics every 10-15s
      - If no HTTP requests arrive, middleware never runs → metrics go stale
      - psutil.cpu_percent(interval=None) returns 0.0 on cold calls
      - This thread ensures metrics are always current regardless of traffic
    """
    # warmup: first call initialises psutil's internal CPU timer
    psutil.cpu_percent(interval=1.0)   # blocking 1s warmup, then loop non-blocking

    while not stop_event.is_set():
        try:
            # CPU — interval=None uses time since last call (fast, accurate)
            cpu = psutil.cpu_percent(interval=None)
            M.CPU_PERCENT.set(cpu)

            # Memory
            vm = psutil.virtual_memory()
            M.MEMORY_PERCENT.set(vm.percent)
            M.MEMORY_USED_GB.set(round(vm.used / 1e9, 3))

            # GPU (only if available) ─
            if torch.cuda.is_available():
                M.GPU_ALLOC.set(torch.cuda.memory_allocated() / 1e9)
                M.GPU_RESERVED.set(torch.cuda.memory_reserved() / 1e9)

            # Uptime
            if app_state["start_time"]:
                M.UPTIME.set(time.time() - app_state["start_time"])

            # MongoDB status 
            db_up = mongo.is_up()
            M.MONGODB_UP.set(1 if db_up else 0)

            # User counts (cheap count queries)
            M.USERS_REGISTERED.set(mongo.get_all_users_count())
            M.USERS_VERIFIED.set(mongo.get_verified_users_count())

        except Exception as e:
            logger.warning(f"Background metrics error: {e}")

        # sleep 5s — fine-grained enough for Prometheus 10-15s scrape interval
        stop_event.wait(timeout=5.0)


# lifespan 
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["start_time"] = time.time()
    logger.info(f"Starting | device={DEVICE} | model={MODEL_PATH}")

    # start background metrics thread BEFORE model loading
    stop_event = app_state["_stop_bg"]
    bg_thread  = threading.Thread(
        target=_background_metrics_loop,
        args=(stop_event,),
        daemon=True,
        name="metrics-collector"
    )
    bg_thread.start()
    logger.info("Background metrics collector started")

    try:
        set_seed(42)
        if MLFLOW_URI:
            mlflow.set_tracking_uri(MLFLOW_URI)

        if MODEL_PATH and MODEL_PATH.startswith("models:/"):
            logger.info(f"Loading model from Registry: {MODEL_PATH}")
            model = mlflow.pytorch.load_model(MODEL_PATH, map_location=DEVICE)
        else:
            logger.info(f"Loading local model from: {MODEL_PATH}")
            model = build_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)

        model.eval()
        model.to(DEVICE)
        app_state["model"]     = model
        app_state["transform"] = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.7635, 0.5461, 0.5705],
                                 [0.1409, 0.1526, 0.1692]),
        ])
        set_app_state(app_state)

        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        M.MODEL_MEMORY.set(param_bytes)
        M.MODEL_LOADED.set(1)
        logger.info(f"Model ready | {param_bytes/1e6:.1f}MB | device={DEVICE}")

        _restore_metrics_from_db()

    except Exception as e:
        M.MODEL_LOADED.set(0)
        logger.error(f"Startup failed: {e}", exc_info=True)

    yield

    # shutdown — signal background thread to stop
    logger.info("Shutting down background metrics thread")
    stop_event.set()
    bg_thread.join(timeout=10)

    app_state["model"] = None
    M.MODEL_LOADED.set(0)
    mongo.close()
    logger.info("Server shut down")


# app 
app = FastAPI(
    title="Skin Disease Detection API",
    description="AI-powered dermoscopy classification with user accounts and Grad-CAM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(predict_router)


# middleware — request-level metrics only (not system stats)
@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    return response


# system routes 
@app.get("/", tags=["System"])
def root():
    return {
        "name"   : "Skin Disease Detection API",
        "version": "2.0.0",
        "docs"   : "/docs",
        "health" : "/health",
        "ready"  : "/ready",
    }


@app.get("/health", tags=["System"])
def health():
    M.REQUEST_COUNT.labels(endpoint="/health", method="GET", status_code=200).inc()
    return {
        "status"  : "ok",
        "uptime_s": round(time.time() - app_state["start_time"], 1),
        "mongodb" : "up" if mongo.is_up() else "down",
        "model"   : "loaded" if app_state["model"] else "not loaded",
    }


@app.get("/ready", tags=["System"])
def ready():
    if app_state["model"] is None:
        M.REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code=503).inc()
        raise HTTPException(status_code=503, detail="Model not ready")
    M.REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code=200).inc()
    return {
        "status"    : "ready",
        "device"    : str(DEVICE),
        "image_size": IMAGE_SIZE,
        "model"     : MODEL_NAME,
    }


@app.get("/system/info", tags=["System"])
def system_info():
    """Live system resource snapshot."""
    # blocking 0.1s interval for accurate single-call reading
    cpu = psutil.cpu_percent(interval=0.1)
    vm  = psutil.virtual_memory()

    info = {
        "cpu_percent"     : cpu,
        "memory_percent"  : vm.percent,
        "memory_used_gb"  : round(vm.used / 1e9, 2),
        "memory_total_gb" : round(vm.total / 1e9, 2),
        "device"          : str(DEVICE),
        "mongodb_up"      : mongo.is_up(),
        "users_registered": mongo.get_all_users_count(),
        "users_verified"  : mongo.get_verified_users_count(),
        "python_version"  : platform.python_version(),
        "uptime_s"        : round(time.time() - app_state["start_time"], 1),
    }
    if torch.cuda.is_available():
        info["gpu_name"]              = torch.cuda.get_device_name(0)
        info["gpu_allocated_gb"]      = round(torch.cuda.memory_allocated() / 1e9, 2)
        info["gpu_memory_reserved_gb"]= round(torch.cuda.memory_reserved()  / 1e9, 2)
        info["gpu_memory_total_gb"]   = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    M.REQUEST_COUNT.labels(endpoint="/system/info", method="GET", status_code=200).inc()
    return info


@app.get("/classes", tags=["Model"])
def get_classes():
    M.REQUEST_COUNT.labels(endpoint="/classes", method="GET", status_code=200).inc()
    return {
        "classes": [
            {
                "code"        : c,
                "display_name": DISEASE_META[c]["display_name"],
                "risk_level"  : DISEASE_META[c]["risk_level"],
            }
            for c in CLASS_NAMES
        ]
    }


@app.get("/metrics", tags=["System"])
def metrics():
    """Prometheus scrape endpoint — returns all registered metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)