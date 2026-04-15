# src/api/main.py
# FastAPI backend — skin disease detection inference server
# Calls MLflow serve and load latest model, then use loaded local model for inference and prediction. 
# choosen this way to obtain gradcam images and preprocessing.
# uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# keep USE_REMOTE_MODEL=false
#

import io
import os
import json
import time
import uuid
import base64
import logging
import platform
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
import mlflow
import torch
import psutil
import numpy as np
import yaml
import httpx
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
from starlette.responses import Response

from src.models.model import build_model
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed

load_dotenv()
logger = get_logger("api")

# ── constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES      = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_SIZE       = int(os.getenv("IMAGE_SIZE", 336))
MODEL_PATH       = os.getenv("MODEL_PATH")
MODEL_NAME       = os.getenv("MODEL_NAME", "efficientnet_b3")
NUM_CLASSES      = int(os.getenv("NUM_CLASSES", 7))
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_SERVER    = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow")
USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "false").lower() == "true"

# ── disease metadata ───────────────────────────────────────────────────────────
DISEASE_META = {
    "mel": {
        "display_name": "Melanoma",
        "risk_level"  : "HIGH",
        "symptoms"    : ["Asymmetric mole", "Irregular border", "Multiple colors", "Diameter > 6mm", "Evolving shape or size"],
        "advisory"    : "Melanoma is the most dangerous form of skin cancer. Immediate consultation with a dermatologist is strongly recommended.",
        "sources"     : [
            {"name": "AAD — Melanoma", "url": "https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma"},
            {"name": "NHS — Melanoma",  "url": "https://www.nhs.uk/conditions/melanoma-skin-cancer/"},
        ],
    },
    "nv": {
        "display_name": "Melanocytic Nevi",
        "risk_level"  : "LOW",
        "symptoms"    : ["Round or oval shape", "Uniform brown or tan color", "Smooth border", "Usually < 6mm"],
        "advisory"    : "Melanocytic nevi (moles) are usually benign. Monitor for changes in size, color, or shape.",
        "sources"     : [{"name": "AAD — Moles", "url": "https://www.aad.org/public/diseases/a-z/moles-overview"}],
    },
    "bcc": {
        "display_name": "Basal Cell Carcinoma",
        "risk_level"  : "MEDIUM",
        "symptoms"    : ["Pearly or waxy bump", "Flat flesh-colored lesion", "Bleeding or scabbing sore"],
        "advisory"    : "Basal cell carcinoma rarely spreads but requires treatment. Consult a dermatologist promptly.",
        "sources"     : [
            {"name": "AAD — BCC", "url": "https://www.aad.org/public/diseases/skin-cancer/types/common/bcc"},
            {"name": "NHS — BCC", "url": "https://www.nhs.uk/conditions/basal-cell-carcinoma/"},
        ],
    },
    "akiec": {
        "display_name": "Actinic Keratosis",
        "risk_level"  : "MEDIUM",
        "symptoms"    : ["Rough dry scaly patch", "Flat to slightly raised patch", "Hard or wart-like surface", "Itching or burning"],
        "advisory"    : "Actinic keratosis is precancerous. Treatment is recommended to prevent progression.",
        "sources"     : [{"name": "AAD — Actinic Keratosis", "url": "https://www.aad.org/public/diseases/a-z/actinic-keratosis-overview"}],
    },
    "bkl": {
        "display_name": "Benign Keratosis",
        "risk_level"  : "LOW",
        "symptoms"    : ["Waxy scaly slightly raised growth", "Brown black or tan color", "Appears stuck on skin"],
        "advisory"    : "Benign keratosis lesions are non-cancerous. Treatment is generally for cosmetic reasons only.",
        "sources"     : [{"name": "AAD — Seborrheic Keratosis", "url": "https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview"}],
    },
    "df": {
        "display_name": "Dermatofibroma",
        "risk_level"  : "LOW",
        "symptoms"    : ["Small hard bump", "Brown to pink color", "Dimples inward when pinched", "Slightly tender"],
        "advisory"    : "Dermatofibromas are benign. No treatment is usually necessary unless causing discomfort.",
        "sources"     : [{"name": "AAD — Dermatofibroma", "url": "https://www.aad.org/public/diseases/a-z/dermatofibroma-overview"}],
    },
    "vasc": {
        "display_name": "Vascular Lesion",
        "risk_level"  : "LOW",
        "symptoms"    : ["Red purple or blue discoloration", "Flat or raised", "May include angiomas or hemangiomas"],
        "advisory"    : "Most vascular lesions are benign. Consult a dermatologist if lesion grows or bleeds.",
        "sources"     : [{"name": "AAD — Vascular Lesions", "url": "https://www.aad.org/public/diseases/a-z/birthmarks-vascular"}],
    },
}

# ── prometheus metrics — user + system + application level ────────────────────
def _make_counter(name, doc, labels):
    try:
        return Counter(name, doc, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def _make_histogram(name, doc, labels, buckets):
    try:
        return Histogram(name, doc, labels, buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def _make_gauge(name, doc, labels=None):
    try:
        return Gauge(name, doc, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def _make_summary(name, doc, labels):
    try:
        return Summary(name, doc, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

# application level
REQUEST_COUNT         = _make_counter("api_requests_total", "Total API requests", ["endpoint", "method", "status_code"])
ERRORS_TOTAL          = _make_counter("errors_total", "Total errors by type", ["error_type", "endpoint"])
IMAGES_PROCESSED      = _make_counter("images_processed_total", "Total images processed", ["status"])
HIGH_RISK_COUNTER     = _make_counter("high_risk_predictions_total", "High risk (melanoma) predictions", [])
PREDICTION_CLASS      = _make_counter("predictions_total", "Predictions by class", ["predicted_class", "risk_level"])
FEEDBACK_COUNTER      = _make_counter("feedback_total", "User feedback votes", ["vote"])

# inference level
INFERENCE_LATENCY     = _make_histogram("inference_latency_seconds", "Model inference latency", ["mode"],
                                        [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
IMAGE_PROCESS_DURATION= _make_histogram("image_processing_duration_seconds", "Image preprocessing time", ["stage"],
                                        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
CONFIDENCE_HISTOGRAM  = _make_histogram("prediction_confidence", "Confidence score distribution", ["predicted_class"],
                                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
REQUEST_MEMORY_USAGE  = _make_histogram("request_memory_usage_mb", "Memory usage per request (MB)", ["endpoint"],
                                        [50, 100, 200, 500, 1000, 2000])
REQUEST_CPU_USAGE     = _make_histogram("request_cpu_usage_percent", "CPU usage per request", ["endpoint"],
                                        [5, 10, 20, 30, 50, 70, 90])
INFERENCE_LATENCY_SUMMARY = _make_summary("inference_latency_summary", "Summary of inference latency", ["mode"])

# system level
MODEL_LOADED          = _make_gauge("model_loaded", "1 if model is loaded and ready", [])
ACTIVE_REQUESTS       = _make_gauge("active_requests", "Currently active requests", ["endpoint"])
CPU_PERCENT           = _make_gauge("system_cpu_percent", "System CPU utilization", [])
MEMORY_PERCENT        = _make_gauge("system_memory_percent", "System memory utilization", [])
GPU_MEMORY_ALLOCATED  = _make_gauge("gpu_memory_allocated_gb", "GPU memory allocated (GB)", [])
GPU_MEMORY_RESERVED   = _make_gauge("gpu_memory_reserved_gb", "GPU memory reserved (GB)", [])
MODEL_MEMORY_BYTES    = _make_gauge("model_memory_bytes", "Estimated model memory bytes", [])
UPTIME_SECONDS        = _make_gauge("api_uptime_seconds", "API server uptime in seconds", [])

# user level
FEEDBACK_POSITIVE_RATE= _make_gauge("feedback_positive_rate", "Rolling positive feedback rate", [])
UNIQUE_PREDICTIONS    = _make_gauge("unique_predictions_total", "Total unique prediction requests", [])

# ── global state ───────────────────────────────────────────────────────────────
app_state = {
    "model"        : None,
    "transform"    : None,
    "start_time"   : None,
    "pred_count"   : 0,
    "feedback_up"  : 0,
    "feedback_down": 0,
}
feedback_store: dict = {}


# ── lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["start_time"] = time.time()
    logger.info(f"Starting API | device={DEVICE} | model={MODEL_PATH}")

    try:
        set_seed(42)
        mlflow.set_tracking_uri(MLFLOW_SERVER)
        # load model weights directly — no MLflow server dependency at startup
        with open("params.yaml") as f:
            tp = yaml.safe_load(f)["train"]

        if MODEL_PATH.startswith("models:/"):
            logger.info(f"Loading model from Registry: {MODEL_PATH}")
            model = mlflow.pytorch.load_model(MODEL_PATH)
        else:
            logger.info(f"Loading local model from: {MODEL_PATH}")
            model = build_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)

        model.eval()
        model.to(DEVICE)
        app_state["model"] = model

        app_state["transform"] = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.7635, 0.5461, 0.5705],
                                 [0.1409, 0.1526, 0.1692]),
        ])

        # estimate model memory
        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        MODEL_MEMORY_BYTES.set(param_bytes)
        MODEL_LOADED.set(1)
        logger.info(f"Model loaded | params={param_bytes/1e6:.1f}MB")

    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(f"Model load failed: {e}", exc_info=True)

    yield

    app_state["model"] = None
    MODEL_LOADED.set(0)
    logger.info("Server shutting down")


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Disease Detection API",
    description="AI-powered dermoscopy classification with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── middleware — update system metrics on every request ───────────────────────
@app.middleware("http")
async def system_metrics_middleware(request: Request, call_next):
    # system resource snapshot per request
    CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    MEMORY_PERCENT.set(psutil.virtual_memory().percent)

    if torch.cuda.is_available():
        GPU_MEMORY_ALLOCATED.set(torch.cuda.memory_allocated() / 1e9)
        GPU_MEMORY_RESERVED.set(torch.cuda.memory_reserved() / 1e9)

    if app_state["start_time"]:
        UPTIME_SECONDS.set(time.time() - app_state["start_time"])

    response = await call_next(request)
    return response


# ── schemas ────────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    prediction_id: str
    vote         : str    # thumbs_up | thumbs_down
    comment      : str = ""


class FeedbackResponse(BaseModel):
    status        : str
    prediction_id : str
    vote          : str


# ── helpers ────────────────────────────────────────────────────────────────────
def get_model():
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app_state["model"]


def preprocess_image(image_bytes: bytes):
    """Preprocess image bytes → (tensor, numpy_float32)."""
    t0 = time.time()
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        np_img  = np.array(resized).astype(np.float32) / 255.0
        IMAGE_PROCESS_DURATION.labels(stage="decode_resize").observe(time.time() - t0)

        t1     = time.time()
        tensor = app_state["transform"](resized).unsqueeze(0).to(DEVICE)
        IMAGE_PROCESS_DURATION.labels(stage="normalize_tensor").observe(time.time() - t1)

        return tensor, np_img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def get_target_layer(model):
    try:
        return model.features[-1]   # efficientnet + convnext
    except AttributeError:
        return model.layer4[-1]     # resnet


def run_gradcam(model, tensor: torch.Tensor, class_idx: int, np_img: np.ndarray) -> str:
    """Generate GradCAM overlay, return base64 PNG string."""
    try:
        model.eval()
        inner        = model.model if hasattr(model, "model") else model
        target_layer = get_target_layer(inner)
        cam          = GradCAM(model=inner, target_layers=[target_layer])
        grayscale    = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])

        if np_img.max() > 1.0:
            np_img = np_img.astype(np.float32) / 255.0

        cam_img  = show_cam_on_image(np_img, grayscale[0], use_rgb=True)
        buf      = io.BytesIO()
        Image.fromarray(cam_img).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.warning(f"GradCAM failed: {e}")
        return ""


def save_gradcam(result_b64):
    try:
        encoded_data = result_b64.split(",")[1]
        img_bytes = base64.b64decode(encoded_data)
        with open("archived/latest_gradcam.png", "wb") as f:
            f.write(img_bytes)
        print("✅Success: Saved latest_gradcam.png to local directory")
    except Exception as e:
        print(f"❌Failed to save local image: {e}")


async def call_mlflow_server(tensor: torch.Tensor) -> np.ndarray:
    """Optional: call mlflow models serve endpoint instead of local model."""
    payload = {"inputs": tensor.cpu().numpy().tolist()}
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.post(f"{MLFLOW_SERVER}/invocations", json=payload)
        res.raise_for_status()
        return np.array(res.json()["predictions"])


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "name"   : "Skin Disease Detection API",
        "version": "1.0.0",
        "docs"   : "/docs",
        "health" : "/health",
        "ready"  : "/ready",
    }


@app.get("/health", tags=["System"])
def health():
    """Liveness probe — always 200 if server is up."""
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status_code=200).inc()
    return {
        "status"   : "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_s" : round(time.time() - app_state["start_time"], 1) if app_state["start_time"] else 0,
    }


@app.get("/ready", tags=["System"])
def ready():
    """Readiness probe — 503 if model not loaded."""
    if app_state["model"] is None:
        REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code=503).inc()
        raise HTTPException(status_code=503, detail="Model not ready")
    REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code=200).inc()
    return {
        "status"    : "ready",
        "device"    : str(DEVICE),
        "image_size": IMAGE_SIZE,
        "model"     : MODEL_NAME,
    }


@app.get("/classes", tags=["Model"])
def get_classes():
    """All supported disease classes with risk levels."""
    REQUEST_COUNT.labels(endpoint="/classes", method="GET", status_code=200).inc()
    return {
        "classes": [
            {
                "code"        : cls,
                "display_name": DISEASE_META[cls]["display_name"],
                "risk_level"  : DISEASE_META[cls]["risk_level"],
            }
            for cls in CLASS_NAMES
        ]
    }


@app.get("/system/info", tags=["System"])
def system_info():
    """System resource snapshot — CPU, memory, GPU."""
    info = {
        "cpu_percent"   : psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        "python_version": platform.python_version(),
        "device"        : str(DEVICE),
    }
    if torch.cuda.is_available():
        info["gpu_name"]              = torch.cuda.get_device_name(0)
        info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        info["gpu_memory_reserved_gb"]  = round(torch.cuda.memory_reserved()  / 1e9, 2)
        info["gpu_memory_total_gb"]     = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...), model=Depends(get_model)):
    """
    Upload skin image → returns prediction + GradCAM + symptoms + advisory.
    All metrics logged to Prometheus automatically.
    """
    prediction_id = str(uuid.uuid4())
    start_time    = time.time()

    # track system snapshot before inference
    mem_before = psutil.virtual_memory().used / 1e6
    cpu_before = psutil.cpu_percent(interval=None)

    ACTIVE_REQUESTS.labels(endpoint="/predict").inc()

    try:
        image_bytes = await file.read()
        tensor, np_img = preprocess_image(image_bytes)
        tensor = tensor.float()

        # ── inference ──────────────────────────────────────────────────────────
        t_infer = time.time()
        if USE_REMOTE_MODEL:
            probs = await call_mlflow_server(tensor)
            probs = torch.softmax(torch.tensor(probs), dim=-1).squeeze().numpy()
        else:
            with torch.no_grad():
                outputs = model(tensor)
                probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        infer_time = time.time() - t_infer
        INFERENCE_LATENCY.labels(mode="single").observe(infer_time)
        INFERENCE_LATENCY_SUMMARY.labels(mode="single").observe(infer_time)

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        meta       = DISEASE_META[pred_class]

        # ── gradcam ────────────────────────────────────────────────────────────
        gradcam_b64 = run_gradcam(model, tensor, pred_idx, np_img)

        if gradcam_b64:
            save_gradcam(gradcam_b64)
        # ── prometheus counters ────────────────────────────────────────────────
        total_latency = time.time() - start_time
        mem_delta     = psutil.virtual_memory().used / 1e6 - mem_before
        cpu_delta     = psutil.cpu_percent(interval=None)

        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=200).inc()
        IMAGES_PROCESSED.labels(status="success").inc()
        PREDICTION_CLASS.labels(predicted_class=pred_class, risk_level=meta["risk_level"]).inc()
        CONFIDENCE_HISTOGRAM.labels(predicted_class=pred_class).observe(confidence)
        REQUEST_MEMORY_USAGE.labels(endpoint="/predict").observe(max(mem_delta, 0))
        REQUEST_CPU_USAGE.labels(endpoint="/predict").observe(cpu_delta)

        app_state["pred_count"] += 1
        UNIQUE_PREDICTIONS.set(app_state["pred_count"])

        if meta["risk_level"] == "HIGH":
            HIGH_RISK_COUNTER.inc()

        logger.info(
            f"predict | id={prediction_id} | class={pred_class} "
            f"| conf={confidence:.3f} | infer={infer_time*1000:.1f}ms | total={total_latency*1000:.1f}ms"
        )

        return {
            "prediction_id"  : prediction_id,
            "predicted_class": pred_class,
            "display_name"   : meta["display_name"],
            "risk_level"     : meta["risk_level"],
            "confidence"     : round(confidence, 4),
            "all_scores"     : {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
            "symptoms"       : meta["symptoms"],
            "advisory"       : meta["advisory"],
            "sources"        : meta["sources"],
            "gradcam_image"  : gradcam_b64,
            "disclaimer"     : "This tool provides AI-assisted preliminary assessment only. It is NOT a substitute for professional medical diagnosis. Please consult a qualified dermatologist.",
            "inference_ms"   : round(infer_time * 1000, 2),
            "total_ms"       : round(total_latency * 1000, 2),
            "timestamp"      : datetime.utcnow().isoformat(),
        }

    except HTTPException:
        IMAGES_PROCESSED.labels(status="failed").inc()
        ERRORS_TOTAL.labels(error_type="http", endpoint="/predict").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=400).inc()
        raise
    except Exception as e:
        IMAGES_PROCESSED.labels(status="failed").inc()
        ERRORS_TOTAL.labels(error_type="internal", endpoint="/predict").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=500).inc()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.labels(endpoint="/predict").dec()


@app.post("/explain", tags=["Inference"])
async def explain(
    file: UploadFile = File(...),
    class_name: str = None,
    model=Depends(get_model)
):
    """GradCAM heatmap for a given image. Optionally target a specific class."""
    ACTIVE_REQUESTS.labels(endpoint="/explain").inc()
    try:
        image_bytes = await file.read()
        tensor, np_img = preprocess_image(image_bytes)

        if class_name and class_name in CLASS_NAMES:
            class_idx = CLASS_NAMES.index(class_name)
        else:
            with torch.no_grad():
                outputs   = model(tensor)
                probs     = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            class_idx = int(np.argmax(probs))

        gradcam_b64 = run_gradcam(model, tensor, class_idx, np_img)
        if gradcam_b64 is not None:
            save_gradcam(gradcam_b64)
        REQUEST_COUNT.labels(endpoint="/explain", method="POST", status_code=200).inc()

        return {
            "target_class" : CLASS_NAMES[class_idx],
            "display_name" : DISEASE_META[CLASS_NAMES[class_idx]]["display_name"],
            "gradcam_image": gradcam_b64,
        }
    except HTTPException:
        REQUEST_COUNT.labels(endpoint="/explain", method="POST", status_code=400).inc()
        ERRORS_TOTAL.labels(error_type="http", endpoint="/explain").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/explain", method="POST", status_code=500).inc()
        ERRORS_TOTAL.labels(error_type="internal", endpoint="/explain").inc()
        logger.error(f"Explain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.labels(endpoint="/explain").dec()


@app.post("/feedback", tags=["Feedback"], response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Collect thumbs up/down feedback. Will be persisted to MongoDB in next phase."""
    if req.vote not in ("thumbs_up", "thumbs_down"):
        REQUEST_COUNT.labels(endpoint="/feedback", method="POST", status_code=422).inc()
        raise HTTPException(status_code=422, detail="vote must be 'thumbs_up' or 'thumbs_down'")

    feedback_store[req.prediction_id] = {
        "prediction_id": req.prediction_id,
        "vote"         : req.vote,
        "comment"      : req.comment,
        "timestamp"    : datetime.utcnow().isoformat(),
    }

    FEEDBACK_COUNTER.labels(vote=req.vote).inc()
    REQUEST_COUNT.labels(endpoint="/feedback", method="POST", status_code=200).inc()

    # update rolling positive rate gauge
    if req.vote == "thumbs_up":
        app_state["feedback_up"] += 1
    else:
        app_state["feedback_down"] += 1
    total = app_state["feedback_up"] + app_state["feedback_down"]
    if total > 0:
        FEEDBACK_POSITIVE_RATE.set(app_state["feedback_up"] / total)

    logger.info(f"Feedback | id={req.prediction_id} | vote={req.vote}")
    return FeedbackResponse(status="recorded", prediction_id=req.prediction_id, vote=req.vote)


@app.get("/feedback/stats", tags=["Feedback"])
def feedback_stats():
    """Thumbs up/down ratio and all feedback records."""
    total       = len(feedback_store)
    thumbs_up   = sum(1 for f in feedback_store.values() if f["vote"] == "thumbs_up")
    thumbs_down = total - thumbs_up
    REQUEST_COUNT.labels(endpoint="/feedback/stats", method="GET", status_code=200).inc()
    return {
        "total"        : total,
        "thumbs_up"    : thumbs_up,
        "thumbs_down"  : thumbs_down,
        "positive_rate": round(thumbs_up / total, 4) if total > 0 else 0.0,
    }


@app.get("/metrics", tags=["System"])
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)