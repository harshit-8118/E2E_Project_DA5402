# src/api/predict.py
# Inference endpoints — /predict, /explain, /feedback

import io
import os
import time
import uuid
import httpx
import base64
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

import torch
import psutil
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.api.deps import get_current_user, get_optional_user
from src.db.mongodb import mongo
from src.utils.logger import get_logger
# prometheus metrics imported from main app_state via module-level dict
from src.api import metrics as M
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
logger = get_logger("predict")

router = APIRouter(tags=["Inference"])

MLFLOW_SERVER    = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow")
USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "false").lower() == "true"
CLASS_NAMES  = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

DISEASE_META = {
    "mel"  : {"display_name":"Melanoma",           "risk_level":"HIGH",
               "symptoms":["Asymmetric mole","Irregular border","Multiple colors","Diameter > 6mm","Evolving shape"],
               "advisory":"Melanoma is the most dangerous skin cancer. Immediate dermatologist consultation recommended.",
               "sources":[{"name":"AAD — Melanoma","url":"https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma"}]},
    "nv"   : {"display_name":"Melanocytic Nevi",   "risk_level":"LOW",
               "symptoms":["Round or oval shape","Uniform color","Smooth border","< 6mm"],
               "advisory":"Usually benign. Monitor for changes in size or color.",
               "sources":[{"name":"AAD — Moles","url":"https://www.aad.org/public/diseases/a-z/moles-overview"}]},
    "bcc"  : {"display_name":"Basal Cell Carcinoma","risk_level":"MEDIUM",
               "symptoms":["Pearly or waxy bump","Flat flesh-colored lesion","Bleeding sore"],
               "advisory":"Rarely spreads but requires treatment. Consult dermatologist promptly.",
               "sources":[{"name":"AAD — BCC","url":"https://www.aad.org/public/diseases/skin-cancer/types/common/bcc"}]},
    "akiec": {"display_name":"Actinic Keratosis",  "risk_level":"MEDIUM",
               "symptoms":["Rough scaly patch","Itching or burning","Flat to slightly raised"],
               "advisory":"Precancerous. Treatment recommended to prevent progression.",
               "sources":[{"name":"AAD — Actinic Keratosis","url":"https://www.aad.org/public/diseases/a-z/actinic-keratosis-overview"}]},
    "bkl"  : {"display_name":"Benign Keratosis",   "risk_level":"LOW",
               "symptoms":["Waxy scaly growth","Brown or tan color","Stuck-on appearance"],
               "advisory":"Non-cancerous. Treatment for cosmetic reasons only.",
               "sources":[{"name":"AAD — Seborrheic Keratosis","url":"https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview"}]},
    "df"   : {"display_name":"Dermatofibroma",     "risk_level":"LOW",
               "symptoms":["Small hard bump","Brown to pink","Dimples when pinched"],
               "advisory":"Benign. Usually no treatment needed.",
               "sources":[{"name":"AAD — Dermatofibroma","url":"https://www.aad.org/public/diseases/a-z/dermatofibroma-overview"}]},
    "vasc" : {"display_name":"Vascular Lesion",    "risk_level":"LOW",
               "symptoms":["Red purple or blue discoloration","Flat or raised"],
               "advisory":"Usually benign. See dermatologist if lesion grows rapidly.",
               "sources":[{"name":"AAD — Vascular Lesions","url":"https://www.aad.org/public/diseases/a-z/birthmarks-vascular"}]},
}

# app_state reference — set from main.py on startup
_app_state = {}


def set_app_state(state: dict):
    global _app_state
    _app_state = state


def get_model():
    if _app_state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _app_state["model"]


def preprocess_image(image_bytes: bytes):
    t0 = time.time()
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_size = _app_state.get("image_size", 224)
        resized = pil_img.resize((image_size, image_size))
        np_img  = np.array(resized).astype(np.float32) / 255.0
        M.IMG_PROC_DURATION.labels(stage="decode_resize").observe(time.time() - t0)
        t1     = time.time()
        tensor = _app_state["transform"](resized).unsqueeze(0).to(_app_state["device"])
        M.IMG_PROC_DURATION.labels(stage="normalize_tensor").observe(time.time() - t1)
        return tensor, np_img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def get_target_layer(model):
    try:    return model.features[-1]
    except: return model.layer4[-1]


async def call_mlflow_server(tensor: torch.Tensor) -> np.ndarray:
    """Optional: call mlflow models serve endpoint instead of local model."""
    payload = {"inputs": tensor.cpu().numpy().tolist()}
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.post(f"{MLFLOW_SERVER}/invocations", json=payload)
        res.raise_for_status()
        return np.array(res.json()["predictions"])


def save_gradcam(result_b64):
    try:
        encoded_data = result_b64.split(",")[1]
        img_bytes = base64.b64decode(encoded_data)
        with open("archived/latest_gradcam.png", "wb") as f:
            f.write(img_bytes)
        print("✅ Success: Saved latest_gradcam.png to local directory")
    except Exception as e:
        print(f"❌ Failed to save local image: {e}")


def run_gradcam(model, tensor, class_idx, np_img) -> str:
    try:
        model.eval()
        inner = model.model if hasattr(model, "model") else model
        cam   = GradCAM(model=inner, target_layers=[get_target_layer(inner)])
        gray  = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])
        if np_img.max() > 1.0:
            np_img = np_img.astype(np.float32) / 255.0
        cam_img = show_cam_on_image(np_img, gray[0], use_rgb=True)
        buf     = io.BytesIO()
        Image.fromarray(cam_img).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"GradCAM failed: {e}")
        return ""


# ── schemas ────────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    prediction_id: str
    vote         : str
    comment      : str = ""

class FeedbackResponse(BaseModel):
    status        : str
    prediction_id : str
    vote          : str
    stored_in     : str


# ── routes ─────────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(
    file        : UploadFile = File(...),
    model                    = Depends(get_model),
    current_user: dict       = Depends(get_current_user),   # auth required
):
    """Upload skin image → prediction + GradCAM. Requires login."""
    prediction_id = str(uuid.uuid4())
    start_time    = time.time()
    mem_before    = psutil.virtual_memory().used / 1e6
    uid           = current_user["uid"]
    username      = current_user["username"]

    M.ACTIVE_REQUESTS.labels(endpoint="/predict").inc()
    try:
        image_bytes    = await file.read()
        tensor, np_img = preprocess_image(image_bytes)
        tensor         = tensor.float()

        # inference
        t_infer = time.time()
        if USE_REMOTE_MODEL:
            probs = await call_mlflow_server(tensor)
            probs = torch.softmax(torch.tensor(probs), dim=-1).squeeze().numpy()
        else:
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
        infer_time = time.time() - t_infer

        M.INFERENCE_LATENCY.labels(mode="single").observe(infer_time)
        M.INFER_SUMMARY.labels(mode="single").observe(infer_time)

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        meta       = DISEASE_META[pred_class]
        all_scores = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

        # gradcam
        gradcam_b64 = run_gradcam(model, tensor, pred_idx, np_img)

        if gradcam_b64:
            # save_gradcam(gradcam_b64)
            pass

        # save image to MongoDB for future retraining
        image_id = mongo.save_image(
            uid=uid, username=username,
            prediction_id=prediction_id,
            image_bytes=image_bytes,
            filename=file.filename or "upload.jpg",
        )

        # save prediction linked to user
        mongo.save_prediction(
            prediction_id=prediction_id, uid=uid, username=username,
            predicted_class=pred_class, confidence=round(confidence, 4),
            risk_level=meta["risk_level"], all_scores=all_scores,
            inference_ms=round(infer_time * 1000, 2),
            image_id=image_id, image_filename=file.filename,
        )

        # rate limiting tracking
        req_count = mongo.log_request(uid, "/predict")
        if req_count >= 3:
            M.USER_RATE_LIMIT_BREACH.labels(uid=uid, username=username).inc()
            logger.warning(f"Rate limit: {username} made {req_count} requests in last hour")

        # prometheus
        total_ms  = (time.time() - start_time) * 1000
        mem_delta = max(psutil.virtual_memory().used / 1e6 - mem_before, 0)

        M.REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=200).inc()
        M.IMAGES_PROCESSED.labels(status="success").inc()
        M.PREDICTION_CLASS.labels(predicted_class=pred_class, risk_level=meta["risk_level"]).inc()
        M.CONFIDENCE_HIST.labels(predicted_class=pred_class).observe(confidence)
        M.REQ_MEMORY.labels(endpoint="/predict").observe(mem_delta)
        M.REQ_CPU.labels(endpoint="/predict").observe(psutil.cpu_percent(interval=None))
        M.USER_PREDICTIONS.labels(username=username).inc()

        _app_state["pred_count"] = _app_state.get("pred_count", 0) + 1
        M.UNIQUE_PREDS.set(_app_state["pred_count"])

        if meta["risk_level"] == "HIGH":
            M.HIGH_RISK_COUNTER.inc()

        logger.info(f"predict | {username} | {pred_class} | conf={confidence:.3f} | {infer_time*1000:.1f}ms")

        return {
            "prediction_id"  : prediction_id,
            "predicted_class": pred_class,
            "display_name"   : meta["display_name"],
            "risk_level"     : meta["risk_level"],
            "confidence"     : round(confidence, 4),
            "all_scores"     : all_scores,
            "symptoms"       : meta["symptoms"],
            "advisory"       : meta["advisory"],
            "sources"        : meta["sources"],
            "gradcam_image"  : gradcam_b64,
            "disclaimer"     : "AI-assisted assessment only. NOT a substitute for professional medical diagnosis.",
            "inference_ms"   : round(infer_time * 1000, 2),
            "total_ms"       : round(total_ms, 2),
            "image_saved"    : image_id is not None,
            "timestamp"      : datetime.utcnow().isoformat(),
        }

    except HTTPException:
        M.IMAGES_PROCESSED.labels(status="failed").inc()
        M.ERRORS_TOTAL.labels(error_type="http", endpoint="/predict").inc()
        M.REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=400).inc()
        raise
    except Exception as e:
        M.IMAGES_PROCESSED.labels(status="failed").inc()
        M.ERRORS_TOTAL.labels(error_type="internal", endpoint="/predict").inc()
        M.REQUEST_COUNT.labels(endpoint="/predict", method="POST", status_code=500).inc()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        M.ACTIVE_REQUESTS.labels(endpoint="/predict").dec()


@router.post("/explain")
async def explain(
    file        : UploadFile = File(...),
    class_name  : str        = None,
    model                    = Depends(get_model),
    current_user: dict       = Depends(get_current_user),
):
    """GradCAM heatmap for a given class. Requires login."""
    M.ACTIVE_REQUESTS.labels(endpoint="/explain").inc()
    try:
        image_bytes    = await file.read()
        tensor, np_img = preprocess_image(image_bytes)
        if class_name and class_name in CLASS_NAMES:
            class_idx = CLASS_NAMES.index(class_name)
        else:
            with torch.no_grad():
                probs     = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
            class_idx = int(np.argmax(probs))
        gradcam_b64 = run_gradcam(model, tensor, class_idx, np_img)
        if gradcam_b64 is not None:
            # save_gradcam(gradcam_b64)
            pass
        M.REQUEST_COUNT.labels(endpoint="/explain", method="POST", status_code=200).inc()
        return {
            "target_class" : CLASS_NAMES[class_idx],
            "display_name" : DISEASE_META[CLASS_NAMES[class_idx]]["display_name"],
            "gradcam_image": gradcam_b64,
        }
    except HTTPException:
        M.REQUEST_COUNT.labels(endpoint="/explain", method="POST", status_code=400).inc()
        M.ERRORS_TOTAL.labels(error_type="http", endpoint="/explain").inc()
        raise
    except Exception as e:
        M.REQUEST_COUNT.labels(endpoint="/explain", method="POST", status_code=500).inc()
        M.ERRORS_TOTAL.labels(error_type="internal", endpoint="/internal").inc()
        logger.error(f"Explain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        M.ACTIVE_REQUESTS.labels(endpoint="/explain").dec()


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    req         : FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit thumbs up/down on a prediction. Requires login."""
    if req.vote not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=422, detail="vote must be 'thumbs_up' or 'thumbs_down'")

    stored = mongo.save_feedback(
        prediction_id=req.prediction_id,
        uid=current_user["uid"],
        username=current_user["username"],
        vote=req.vote,
        comment=req.comment,
    )

    M.FEEDBACK_COUNTER.labels(vote=req.vote).inc()
    M.REQUEST_COUNT.labels(endpoint="/feedback", method="POST", status_code=200).inc()

    # update rolling rate gauge
    fb_stats = mongo.get_feedback_stats()
    if fb_stats.get("total", 0) > 0:
        M.FEEDBACK_RATE.set(fb_stats["positive_rate"])

    return FeedbackResponse(
        status="recorded", prediction_id=req.prediction_id,
        vote=req.vote, stored_in="mongodb" if stored else "memory",
    )


@router.get("/feedback/stats")
def feedback_stats(current_user: dict = Depends(get_current_user)):
    M.REQUEST_COUNT.labels(endpoint="/feedback/stats", method="GET", status_code=200).inc()
    return mongo.get_feedback_stats()


@router.get("/predictions/stats")
def prediction_stats(current_user: dict = Depends(get_current_user)):
    M.REQUEST_COUNT.labels(endpoint="/predictions/stats", method="GET", status_code=200).inc()
    return mongo.get_prediction_stats()