# src/api/main.py
# FastAPI app entry point — wires auth, predict, metrics routers
# uvicorn src.api.main:app --host 0.0.0.0 --port 8000

import os
import time
import platform
from contextlib import asynccontextmanager

import torch
import psutil
import yaml
import mlflow
from dotenv import load_dotenv
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

load_dotenv()
logger = get_logger("api")

IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE", 224))
MODEL_PATH  = os.getenv("MODEL_PATH", "outputs/models/best_model.pth")
MODEL_NAME  = os.getenv("MODEL_NAME", "efficientnet_b3")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 7))
DEVICE      = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "")

app_state = {
    "model": None, "transform": None,
    "start_time": None, "pred_count": 0,
    "device": DEVICE, "image_size": IMAGE_SIZE,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["start_time"] = time.time()
    logger.info(f"Starting | device={DEVICE} | model={MODEL_PATH}")
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
            transforms.Normalize([0.7635, 0.5461, 0.5705], [0.1409, 0.1526, 0.1692]),
        ])
        set_app_state(app_state)

        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        M.MODEL_MEMORY.set(param_bytes)
        M.MODEL_LOADED.set(1)
        M.MONGODB_UP.set(1 if mongo.is_up() else 0)
        M.USERS_REGISTERED.set(mongo.get_all_users_count())
        M.USERS_VERIFIED.set(mongo.get_verified_users_count())
        logger.info(f"Model ready | {param_bytes/1e6:.1f}MB")
    except Exception as e:
        M.MODEL_LOADED.set(0)
        logger.error(f"Startup failed: {e}", exc_info=True)
    yield
    app_state["model"] = None
    M.MODEL_LOADED.set(0)
    mongo.close()
    logger.info("Server shut down")


app = FastAPI(
    title="Skin Disease Detection API",
    description="AI-powered dermoscopy classification with user accounts and Grad-CAM",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(auth_router)
app.include_router(predict_router)


@app.middleware("http")
async def system_metrics_middleware(request: Request, call_next):
    M.CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    vm = psutil.virtual_memory()
    M.MEMORY_PERCENT.set(vm.percent)
    M.MEMORY_USED_GB.set(round(vm.used / 1e9, 2))
    if torch.cuda.is_available():
        M.GPU_ALLOC.set(torch.cuda.memory_allocated() / 1e9)
        M.GPU_RESERVED.set(torch.cuda.memory_reserved() / 1e9)
    if app_state["start_time"]:
        M.UPTIME.set(time.time() - app_state["start_time"])
    M.MONGODB_UP.set(1 if mongo.is_up() else 0)
    M.USERS_REGISTERED.set(mongo.get_all_users_count())
    M.USERS_VERIFIED.set(mongo.get_verified_users_count())
    return await call_next(request)


@app.get("/", tags=["System"])
def root():
    return {"name"   : "Skin Disease Detection API",
        "version": "2.0.0",
        "docs"   : "/docs",
        "health" : "/health",
        "ready"  : "/ready",}


@app.get("/health", tags=["System"])
def health():
    M.REQUEST_COUNT.labels(endpoint="/health", method="GET", status_code=200).inc()
    return {
        "status" : "ok",
        "uptime_s": round(time.time() - app_state["start_time"], 1),
        "mongodb" : "up" if mongo.is_up() else "down",
        "model"  : "loaded" if app_state["model"] else "not loaded",
    }


@app.get("/ready", tags=["System"])
def ready():
    if app_state["model"] is None:
        M.REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code=503).inc()
        raise HTTPException(status_code=503, detail="Model not ready")
    M.REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code=200).inc()
    return {"status": "ready", "device": str(DEVICE), "image_size": IMAGE_SIZE, "model" : MODEL_NAME,}


@app.get("/system/info", tags=["System"])
def system_info():
    info = {
        "cpu_percent"     : psutil.cpu_percent(interval=0.1),
        "memory_percent"  : psutil.virtual_memory().percent,
        "memory_used_gb"  : round(psutil.virtual_memory().used / 1e9, 2),
        "device"          : str(DEVICE),
        "mongodb_up"      : mongo.is_up(),
        "users_registered": mongo.get_all_users_count(),
        "users_verified"  : mongo.get_verified_users_count(),
        "python_version": platform.python_version(),
    }
    if torch.cuda.is_available():
        info["gpu_name"]       = torch.cuda.get_device_name(0)
        info["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        info["gpu_memory_reserved_gb"]  = round(torch.cuda.memory_reserved()  / 1e9, 2)
        info["gpu_memory_total_gb"]     = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    M.REQUEST_COUNT.labels(endpoint="/system/info", method="GET", status_code=200).inc()
    return info


@app.get("/classes", tags=["Model"])
def get_classes():
    M.REQUEST_COUNT.labels(endpoint="/classes", method="GET", status_code=200).inc()
    return {"classes": [{"code": c, "display_name": DISEASE_META[c]["display_name"],
                         "risk_level": DISEASE_META[c]["risk_level"]} for c in CLASS_NAMES]}


@app.get("/metrics", tags=["System"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)