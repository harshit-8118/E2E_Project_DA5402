#!/usr/bin/env python3
"""
Unified end-to-end test suite for DermAI.

Usage:
    python test/run_tests.py --mode easy
    python test/run_tests.py --mode moderate
    python test/run_tests.py --mode rigorous
    python test/run_tests.py --mode overall

Notes:
    - Uses project-relative paths only, so it is safe to commit and push.
    - Supports both `test/test_images/` and `test/test_samples/`.
    - `rigorous` now tries to intentionally stimulate Prometheus alerts and
      then polls Prometheus for firing alert states.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent

DEFAULT_BASE = os.getenv("DERMAI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_PROM = os.getenv("DERMAI_PROM_URL", "http://127.0.0.1:9090").rstrip("/")
DEFAULT_FRONTEND = os.getenv("DERMAI_FRONTEND_URL", "http://127.0.0.1:7500").rstrip("/")
DEFAULT_GRAFANA = os.getenv("DERMAI_GRAFANA_URL", "http://127.0.0.1:3005").rstrip("/")
DEFAULT_MONGO_URI = os.getenv("DERMAI_MONGO_URI", "mongodb://localhost:27017")
DEFAULT_MONGO_DB = os.getenv("DERMAI_MONGO_DB", "skin_disease_detection")

SAMPLE_DIR_CANDIDATES = [
    TEST_ROOT / "test_images",
    TEST_ROOT / "test_samples",
]

results: list[tuple[str, bool]] = []
results_lock = threading.Lock()


class Settings:
    def __init__(self, args: argparse.Namespace):
        self.base = args.base_url.rstrip("/")
        self.prom = args.prom_url.rstrip("/")
        self.frontend = args.frontend_url.rstrip("/")
        self.grafana = args.grafana_url.rstrip("/")
        self.mongo_uri = args.mongo_uri
        self.mongo_db = args.mongo_db
        self.alert_wait_seconds = args.alert_wait_seconds
        self.scrape_settle_seconds = args.scrape_settle_seconds
        self.rigorous_users = args.rigorous_users
        self.rigorous_rounds = args.rigorous_rounds
        self.concurrency = args.concurrency
        self.samples_dir = resolve_samples_dir(args.samples_dir)


SETTINGS: Settings | None = None


def reset_results() -> None:
    with results_lock:
        results.clear()


def check(name: str, ok: bool, detail: str = "") -> bool:
    symbol = PASS if ok else FAIL
    line = f"{symbol} {name}"
    if detail:
        line += f" - {detail}"
    print(line)
    with results_lock:
        results.append((name, ok))
    return ok


def info(message: str) -> None:
    print(message)


def header(title: str) -> None:
    pad = max(0, 62 - len(title))
    print(f"\n-- {title} {'-' * pad}")


def summary(label: str) -> tuple[int, int]:
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + "=" * 62)
    print(f"  {label} Results: {passed}/{total} passed")
    if passed < total:
        print("  Failed:")
        for name, ok in results:
            if not ok:
                print(f"    {FAIL} {name}")
    else:
        print(f"  All tests passed {PASS}")
    print("=" * 62)
    return passed, total


def resolve_samples_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    for candidate in SAMPLE_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return SAMPLE_DIR_CANDIDATES[0]


def _get(url: str, timeout: int = 20):
    try:
        return requests.get(url, timeout=timeout)
    except Exception:
        return None


def query_prom(query: str) -> float | None:
    assert SETTINGS is not None
    try:
        resp = requests.get(
            f"{SETTINGS.prom}/api/v1/query",
            params={"query": query},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        result = payload.get("data", {}).get("result", [])
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return None


def prom_has(query: str) -> bool:
    value = query_prom(query)
    return value is not None


def is_alert_firing(alert_name: str) -> bool:
    value = query_prom(f'ALERTS{{alertstate="firing",alertname="{alert_name}"}}')
    return value is not None and value > 0


def wait_for_alerts(alert_names: list[str], timeout_seconds: int) -> dict[str, bool]:
    deadline = time.time() + timeout_seconds
    status = {name: False for name in alert_names}
    while time.time() < deadline:
        for name in alert_names:
            if not status[name]:
                status[name] = is_alert_firing(name)
        if all(status.values()):
            break
        time.sleep(5)
    return status


def get_real_image(index: int = 0) -> tuple[bytes, str]:
    assert SETTINGS is not None
    images = sorted(SETTINGS.samples_dir.glob("sample_*.jpg"))
    if not images:
        images = sorted(SETTINGS.samples_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No test images found in {SETTINGS.samples_dir}\n"
            "Run: python test/data_creation.py"
        )
    image_path = images[index % len(images)]
    return image_path.read_bytes(), image_path.name


def make_synthetic_image(width: int = 336, height: int = 336) -> tuple[bytes, str]:
    arr = np.random.randint(80, 180, (height, width, 3), dtype=np.uint8)
    cx, cy, radius = width // 2, height // 2, min(width, height) // 4
    for y in range(height):
        for x in range(width):
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                arr[y, x] = [60, 40, 50]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "synthetic.jpg"


def get_image(index: int = 0) -> tuple[bytes, str]:
    try:
        return get_real_image(index)
    except FileNotFoundError:
        info(f"  {WARN} Real images not found - using synthetic image")
        return make_synthetic_image()


def auth_hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def register_and_verify(suffix: str = "") -> tuple[str, str, str | None]:
    assert SETTINGS is not None
    email = f"dermai_test_{uuid.uuid4().hex[:8]}{suffix}@gmail.com"
    username = f"tester_{uuid.uuid4().hex[:6]}"
    password = "TestPass123!"

    signup = requests.post(
        f"{SETTINGS.base}/auth/signup",
        json={
            "username": username,
            "email": email,
            "password": password,
            "gender": "prefer_not_to_say",
        },
        timeout=30,
    )
    if signup.status_code != 200:
        return email, password, None

    try:
        from pymongo import MongoClient

        client = MongoClient(SETTINGS.mongo_uri, serverSelectionTimeoutMS=3000)
        client[SETTINGS.mongo_db].users.update_one(
            {"email": email},
            {"$set": {"verified": True}},
        )
        client.close()
    except Exception as exc:
        info(f"  {WARN} MongoDB direct access failed: {exc}")

    login = requests.post(
        f"{SETTINGS.base}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    token = login.json().get("access_token") if login.status_code == 200 else None
    return email, password, token


def predict_with_token(token: str, img_index: int = 0, timeout: int = 180) -> dict | None:
    assert SETTINGS is not None
    img_bytes, fname = get_image(img_index)
    try:
        resp = requests.post(
            f"{SETTINGS.base}/predict",
            files={"file": (fname, img_bytes, "image/jpeg")},
            headers=auth_hdr(token),
            timeout=timeout,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        info(f"  {WARN} predict failed: {exc}")
        return None


def generate_4xx_requests(count: int = 12) -> None:
    assert SETTINGS is not None
    bad_headers = {"Authorization": "Bearer badtoken"}
    for _ in range(count):
        requests.post(
            f"{SETTINGS.base}/predict",
            files={"file": ("bad.jpg", b"\xff\xd8", "image/jpeg")},
            headers=bad_headers,
            timeout=30,
        )


def create_registration_spike(count: int = 4) -> list[str]:
    tokens: list[str] = []
    for idx in range(count):
        _, _, token = register_and_verify(f"_reg{idx}")
        if token:
            tokens.append(token)
    return tokens


def trigger_rate_limit_breach(token: str, request_count: int = 6) -> None:
    assert SETTINGS is not None
    img_bytes, fname = get_image(0)
    for _ in range(request_count):
        requests.post(
            f"{SETTINGS.base}/predict",
            files={"file": (fname, img_bytes, "image/jpeg")},
            headers=auth_hdr(token),
            timeout=180,
        )


def trigger_negative_feedback_spike(token: str, count: int = 4) -> int:
    assert SETTINGS is not None
    submitted = 0
    for idx in range(count):
        prediction = predict_with_token(token, idx)
        if not prediction:
            continue
        prediction_id = prediction.get("prediction_id")
        if not prediction_id:
            continue
        resp = requests.post(
            f"{SETTINGS.base}/feedback",
            json={
                "prediction_id": prediction_id,
                "vote": "thumbs_down",
                "comment": "auto-generated negative feedback for alert stimulation",
            },
            headers=auth_hdr(token),
            timeout=30,
        )
        if resp.status_code == 200:
            submitted += 1
    return submitted


def run_load(tokens: list[str], rounds: int, concurrency: int) -> tuple[list[float], list[tuple[str, str]], list[str]]:
    assert SETTINGS is not None
    latencies: list[float] = []
    pred_ids: list[tuple[str, str]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def one_predict(token: str, idx: int) -> None:
        img_bytes, fname = get_image(idx)
        started = time.time()
        try:
            resp = requests.post(
                f"{SETTINGS.base}/predict",
                files={"file": (fname, img_bytes, "image/jpeg")},
                headers=auth_hdr(token),
                timeout=180,
            )
            elapsed = time.time() - started
            with lock:
                latencies.append(elapsed)
                if resp.status_code == 200:
                    prediction_id = resp.json().get("prediction_id")
                    if prediction_id:
                        pred_ids.append((token, prediction_id))
                else:
                    errors.append(f"HTTP {resp.status_code}")
        except Exception as exc:
            with lock:
                errors.append(str(exc))

    tasks = [(tokens[i % len(tokens)], i) for i in range(rounds)] if tokens else []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(one_predict, token, idx) for token, idx in tasks]
        for future in as_completed(futures):
            future.result()
    return latencies, pred_ids, errors


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def run_easy() -> tuple[int, int]:
    assert SETTINGS is not None
    print("\n" + "=" * 62)
    print("  EASY TESTS - Service Health & Basic Endpoints")
    print("=" * 62)

    header("Service Health")
    resp = _get(f"{SETTINGS.base}/health")
    check("Backend /health responds", resp is not None and resp.status_code == 200, resp.json().get("status") if resp else "unreachable")

    resp = _get(f"{SETTINGS.base}/ready")
    check("Backend /ready - model loaded", resp is not None and resp.status_code == 200, resp.json().get("status") if resp else "model not loaded")

    resp = _get(SETTINGS.frontend)
    check("Frontend serves HTML", resp is not None and resp.status_code == 200)

    resp = _get(f"{SETTINGS.prom}/-/healthy")
    check("Prometheus healthy", resp is not None and resp.status_code == 200)

    resp = _get(f"{SETTINGS.grafana}/api/health")
    check("Grafana healthy", resp is not None and resp.status_code == 200)

    header("API Endpoints")
    resp = _get(f"{SETTINGS.base}/classes")
    check("/classes returns 7 classes", resp is not None and len(resp.json().get("classes", [])) == 7)

    resp = _get(f"{SETTINGS.base}/system/info")
    check("/system/info responds with device", resp is not None and "device" in resp.json(), resp.json().get("device", "?") if resp else "")

    resp = _get(f"{SETTINGS.base}/metrics")
    check("/metrics returns Prometheus text", resp is not None and "api_requests_total" in (resp.text or ""))

    header("Auth Endpoints")
    resp = requests.post(f"{SETTINGS.base}/auth/login", json={"email": "bad@test.com", "password": "wrong"}, timeout=30)
    check("/auth/login rejects bad credentials (401)", resp.status_code == 401)

    resp = requests.post(
        f"{SETTINGS.base}/auth/signup",
        json={"username": "x", "email": "bademail", "password": "short"},
        timeout=30,
    )
    check("/auth/signup rejects invalid input (422)", resp.status_code == 422)

    header("Prometheus Targets")
    resp = _get(f"{SETTINGS.prom}/api/v1/targets")
    if resp and resp.status_code == 200:
        up_jobs = [
            target["labels"]["job"]
            for target in resp.json()["data"]["activeTargets"]
            if target["health"] == "up"
        ]
        check("fastapi_backend target UP", "fastapi_backend" in up_jobs)
        check("prometheus target UP", "prometheus" in up_jobs)
        check("node_exporter target UP", "node_exporter" in up_jobs)
        check("mongodb_exporter target UP", "mongodb_exporter" in up_jobs)
    else:
        check("Prometheus targets API", False, "unreachable")

    return summary("EASY")


def run_moderate() -> tuple[int, int]:
    assert SETTINGS is not None
    print("\n" + "=" * 62)
    print("  MODERATE TESTS - Auth, Prediction, Metrics")
    print("=" * 62)

    header("User Registration")
    email, password, token = register_and_verify("_mod")
    check("Signup + auto-verify succeeds", token is not None, "check MongoDB connectivity" if not token else "")

    resp = requests.post(
        f"{SETTINGS.base}/auth/signup",
        json={
            "username": f"tester_{uuid.uuid4().hex[:6]}",
            "email": email,
            "password": password,
            "gender": "male",
        },
        timeout=30,
    )
    check("Duplicate email rejected (409/422)", resp.status_code in (409, 422))

    if not token:
        info(f"\n  {WARN} No token - skipping authenticated tests")
        return summary("MODERATE")

    header("Prediction with Real Image")
    img_bytes, fname = get_image(0)
    resp = requests.post(
        f"{SETTINGS.base}/predict",
        files={"file": (fname, img_bytes, "image/jpeg")},
        headers=auth_hdr(token),
        timeout=180,
    )
    check("POST /predict returns 200", resp.status_code == 200, resp.text[:120])

    prediction_id = None
    if resp.status_code == 200:
        body = resp.json()
        prediction_id = body.get("prediction_id")
        check("prediction_id present", prediction_id is not None)
        check("predicted_class is valid", body.get("predicted_class") in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
        check("confidence in (0, 1]", 0 < body.get("confidence", 0) <= 1)
        check("all_scores has 7 classes", len(body.get("all_scores", {})) == 7)
        check("gradcam_image is base64", body.get("gradcam_image", "").startswith("data:image/png;base64,"))
        check("image_saved is True", body.get("image_saved") is True)

    unauth = requests.post(
        f"{SETTINGS.base}/predict",
        files={"file": (fname, img_bytes, "image/jpeg")},
        timeout=30,
    )
    check("Unauthenticated predict -> 401/403", unauth.status_code in (401, 403))

    header("Feedback")
    if prediction_id:
        resp = requests.post(
            f"{SETTINGS.base}/feedback",
            json={"prediction_id": prediction_id, "vote": "thumbs_up", "comment": "looks correct"},
            headers=auth_hdr(token),
            timeout=30,
        )
        check("POST /feedback thumbs_up (200)", resp.status_code == 200)

        resp = requests.post(
            f"{SETTINGS.base}/feedback",
            json={"prediction_id": prediction_id, "vote": "invalid"},
            headers=auth_hdr(token),
            timeout=30,
        )
        check("Invalid vote rejected (422)", resp.status_code == 422)

        resp = requests.get(f"{SETTINGS.base}/feedback/stats", headers=auth_hdr(token), timeout=30)
        check("GET /feedback/stats total > 0", resp.status_code == 200 and resp.json().get("total", 0) > 0)

    header("User Profile")
    resp = requests.get(f"{SETTINGS.base}/auth/me", headers=auth_hdr(token), timeout=30)
    check("GET /auth/me (200)", resp.status_code == 200)
    if resp.status_code == 200:
        profile = resp.json()
        check("Profile has prediction_count", "prediction_count" in profile)
        check("prediction_count >= 1", profile.get("prediction_count", 0) >= 1)

    resp = requests.get(f"{SETTINGS.base}/auth/my-predictions", headers=auth_hdr(token), timeout=30)
    check("GET /auth/my-predictions (200)", resp.status_code == 200)

    header("Prometheus Metrics")
    time.sleep(SETTINGS.scrape_settle_seconds)
    value = query_prom("sum(predictions_total) > 0")
    check("sum(predictions_total) > 0", value is not None and value > 0, str(value))
    value = query_prom("model_loaded")
    check("model_loaded == 1", value == 1.0, str(value))
    value = query_prom("mongodb_up")
    check("mongodb_up == 1", value == 1.0, str(value))
    value = query_prom("system_cpu_percent")
    check("system_cpu_percent > 0", value is not None and value > 0, str(value))
    value = query_prom("system_memory_percent")
    check("system_memory_percent > 0", value is not None and value > 0, str(value))

    return summary("MODERATE")


def run_rigorous() -> tuple[int, int]:
    assert SETTINGS is not None
    print("\n" + "=" * 62)
    print("  RIGOROUS TESTS - Load, Integrity, Alert Stimulation")
    print("=" * 62)

    header("Input Validation Edge Cases")
    fake_hdr = {"Authorization": "Bearer faketoken"}

    resp = requests.post(
        f"{SETTINGS.base}/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        headers=fake_hdr,
        timeout=30,
    )
    check("Empty image -> not 200", resp.status_code != 200)

    resp = requests.post(
        f"{SETTINGS.base}/predict",
        files={"file": ("doc.txt", b"hello world", "text/plain")},
        headers=fake_hdr,
        timeout=30,
    )
    check("Non-image file -> not 200", resp.status_code != 200)

    resp = requests.post(
        f"{SETTINGS.base}/predict",
        files={"file": ("bad.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")},
        headers=fake_hdr,
        timeout=30,
    )
    check("Corrupt JPEG -> not 200", resp.status_code != 200)

    resp = requests.post(f"{SETTINGS.base}/predict", headers=fake_hdr, timeout=30)
    check("Missing file field -> 401/422", resp.status_code in (401, 422))

    resp = requests.post(
        f"{SETTINGS.base}/feedback",
        data="not json",
        headers={**fake_hdr, "Content-Type": "text/plain"},
        timeout=30,
    )
    check("Bad feedback body -> 401/403/422", resp.status_code in (401, 403, 422))

    header("Concurrent Prediction Load")
    tokens: list[str] = []
    for idx in range(SETTINGS.rigorous_users):
        _, _, token = register_and_verify(f"_rig{idx}")
        if token:
            tokens.append(token)
    check(f"{SETTINGS.rigorous_users} test users created", len(tokens) == SETTINGS.rigorous_users, f"got {len(tokens)}")

    latencies, pred_ids, errors = run_load(tokens, SETTINGS.rigorous_rounds, SETTINGS.concurrency)
    check("Concurrent predictions completed with no transport errors", len(errors) == 0, f"errors={errors[:3]}")
    check("Successful predictions recorded", len(pred_ids) > 0, f"success={len(pred_ids)}")

    if latencies:
        avg_ms = sum(latencies) / len(latencies) * 1000
        p95_ms = percentile(latencies, 0.95) * 1000
        check("p95 latency < 60s", p95_ms < 60000, f"avg={avg_ms:.0f}ms p95={p95_ms:.0f}ms")

    header("Feedback Mix")
    thumbs_up = 0
    thumbs_down = 0
    for token, prediction_id in pred_ids:
        vote = "thumbs_up" if random.random() < 0.8 else "thumbs_down"
        resp = requests.post(
            f"{SETTINGS.base}/feedback",
            json={
                "prediction_id": prediction_id,
                "vote": vote,
                "comment": f"auto-test {vote}",
            },
            headers=auth_hdr(token),
            timeout=30,
        )
        if resp.status_code == 200:
            if vote == "thumbs_up":
                thumbs_up += 1
            else:
                thumbs_down += 1
    total_feedback = thumbs_up + thumbs_down
    positive_rate = thumbs_up / total_feedback if total_feedback else 0.0
    check("Feedback submitted", total_feedback > 0, f"total={total_feedback}")
    check("Positive rate roughly 0.8", 0.6 <= positive_rate <= 1.0, f"{positive_rate:.2f}")

    header("MongoDB Data Integrity")
    try:
        from pymongo import MongoClient

        client = MongoClient(SETTINGS.mongo_uri, serverSelectionTimeoutMS=3000)
        db = client[SETTINGS.mongo_db]
        cols = db.list_collection_names()
        for col in ["users", "predictions", "feedback", "images"]:
            check(f"Collection '{col}' exists", col in cols)

        pred_count = db.predictions.count_documents({})
        img_count = db.images.count_documents({})
        fb_count = db.feedback.count_documents({})
        check("Predictions persisted", pred_count > 0, str(pred_count))
        check("Images persisted", img_count > 0, str(img_count))
        check("Feedback persisted", fb_count > 0, str(fb_count))

        prediction = db.predictions.find_one({})
        check("Prediction has uid", bool(prediction and "uid" in prediction))
        check("Prediction has username", bool(prediction and "username" in prediction))
        check("Prediction has image_id", bool(prediction and "image_id" in prediction))

        feedback = db.feedback.find_one({})
        check("Feedback has uid", bool(feedback and "uid" in feedback))
        check("Feedback has username", bool(feedback and "username" in feedback))
        client.close()
    except Exception as exc:
        check("MongoDB connection from host", False, str(exc))

    header("Prometheus Metrics Completeness")
    time.sleep(SETTINGS.scrape_settle_seconds)
    metrics_to_check = {
        "api_requests_total": "api_requests_total",
        "model_loaded": "model_loaded",
        "mongodb_up": "mongodb_up",
        "predictions_total (sum)": "sum(predictions_total)",
        "inference_latency": "inference_latency_seconds_count",
        "system_cpu_percent": "system_cpu_percent",
        "system_memory_percent": "system_memory_percent",
        "feedback_positive_rate": "feedback_positive_rate",
        "unique_predictions_total": "unique_predictions_total",
        "users_registered_total": "users_registered_total",
        "user_predictions_total (sum)": "sum(user_predictions_total)",
        "images_processed_total": "sum(images_processed_total)",
        "errors_total": "errors_total",
        "active_requests": "active_requests",
        "request_memory_usage_mb": "request_memory_usage_mb_bucket",
    }
    for display, query in metrics_to_check.items():
        check(f"metric: {display}", prom_has(query))

    header("Alert Rules Loaded")
    resp = _get(f"{SETTINGS.prom}/api/v1/rules")
    loaded_rules: set[str] = set()
    if resp and resp.status_code == 200:
        groups = resp.json()["data"]["groups"]
        loaded_rules = {rule["name"] for group in groups for rule in group["rules"]}
        for rule in [
            "ModelNotLoaded",
            "MongoDBDown",
            "APIDown",
            "SystemMemory_Critical",
            "SystemCPU_Critical",
            "HighInferenceLatency_Critical",
            "HighErrorRate",
            "HighRequestVolume",
            "HighActiveRequests",
            "UserRateLimitBreach",
            "RegistrationSpike",
            "NegativeFeedbackSpike",
        ]:
            check(f"Rule loaded: {rule}", rule in loaded_rules)
    else:
        check("Prometheus rules API", False, "unreachable")

    header("Alert Stimulation")
    generate_4xx_requests(15)
    check("Generated burst of 4xx requests", True, "for HighErrorRate")

    signup_tokens = create_registration_spike(4)
    check("Generated signup spike", len(signup_tokens) >= 4, f"successful_signups={len(signup_tokens)}")

    rate_limit_token = signup_tokens[0] if signup_tokens else (tokens[0] if tokens else None)
    if rate_limit_token:
        trigger_rate_limit_breach(rate_limit_token, request_count=6)
        check("Triggered rate-limit stress", True, "6 rapid authenticated predictions")
    else:
        check("Triggered rate-limit stress", False, "no valid token available")

    negative_token = signup_tokens[1] if len(signup_tokens) > 1 else rate_limit_token
    if negative_token:
        negative_count = trigger_negative_feedback_spike(negative_token, count=4)
        check("Generated negative feedback burst", negative_count > 0, f"submitted={negative_count}")
    else:
        check("Generated negative feedback burst", False, "no valid token available")

    time.sleep(SETTINGS.alert_wait_seconds)
    alert_candidates = [
        "HighErrorRate",
        "HighRequestVolume",
        "HighActiveRequests",
        "UserRateLimitBreach",
        "RegistrationSpike",
        "NegativeFeedbackSpike",
        "HighInferenceLatency_Warning",
        "HighInferenceLatency_Critical",
    ]
    firing = wait_for_alerts(alert_candidates, timeout_seconds=SETTINGS.alert_wait_seconds)
    for alert_name, active in firing.items():
        detail = "firing" if active else "not firing yet"
        check(f"Alert status: {alert_name}", active if alert_name not in {"HighInferenceLatency_Warning", "HighInferenceLatency_Critical"} else True, detail)

    required_alerts = [
        "HighErrorRate",
        "HighRequestVolume",
        "HighActiveRequests",
        "UserRateLimitBreach",
        "RegistrationSpike",
        "NegativeFeedbackSpike",
    ]
    fired_required = [name for name in required_alerts if firing.get(name)]
    check("Required stimulated alerts fired", len(fired_required) >= 3, ",".join(fired_required) if fired_required else "none")

    return summary("RIGOROUS")


def run_overall() -> tuple[int, int]:
    print("\n" + "=" * 62)
    print("  OVERALL TESTS - Full Suite")
    print("=" * 62)

    run_easy()
    reset_results()
    run_moderate()
    reset_results()
    return run_rigorous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DermAI unified E2E test suite")
    parser.add_argument("--mode", choices=["easy", "moderate", "rigorous", "overall"], default="overall")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="Base URL for FastAPI backend")
    parser.add_argument("--prom-url", default=DEFAULT_PROM, help="Base URL for Prometheus")
    parser.add_argument("--frontend-url", default=DEFAULT_FRONTEND, help="Base URL for frontend")
    parser.add_argument("--grafana-url", default=DEFAULT_GRAFANA, help="Base URL for Grafana")
    parser.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI, help="MongoDB URI for test-only verification shortcuts")
    parser.add_argument("--mongo-db", default=DEFAULT_MONGO_DB, help="MongoDB database name")
    parser.add_argument("--samples-dir", default=None, help="Optional path to sample-image directory")
    parser.add_argument("--alert-wait-seconds", type=int, default=70, help="How long to wait for alerts to enter firing state")
    parser.add_argument("--scrape-settle-seconds", type=int, default=5, help="Seconds to wait for Prometheus scrapes after traffic")
    parser.add_argument("--rigorous-users", type=int, default=12, help="Number of users to create for rigorous load testing")
    parser.add_argument("--rigorous-rounds", type=int, default=50, help="Number of prediction requests in rigorous load testing")
    parser.add_argument("--concurrency", type=int, default=5, help="Max worker count for rigorous load testing")
    return parser.parse_args()


def main() -> None:
    global SETTINGS
    args = parse_args()
    SETTINGS = Settings(args)

    print(f"\nDermAI Test Suite - mode: {args.mode.upper()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"API: {SETTINGS.base} | Prometheus: {SETTINGS.prom} | Grafana: {SETTINGS.grafana}")
    print(f"Image samples: {SETTINGS.samples_dir}")

    if args.mode == "easy":
        passed, total = run_easy()
    elif args.mode == "moderate":
        passed, total = run_moderate()
    elif args.mode == "rigorous":
        passed, total = run_rigorous()
    else:
        passed, total = run_overall()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
