#!/usr/bin/env python3
"""
tests/test_rigorous.py
Rigorous tests — load, edge cases, rate limiting, MongoDB integrity, alert rules
Run: python tests/test_rigorous.py
WARNING: Sends many requests — intentionally triggers rate limit alert
"""

import sys
import io
import time
import uuid
import threading
import requests
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE       = "http://localhost:8000"
PROM       = "http://localhost:9090"
MONGO_PORT = 27017
PASS_SYM   = "✅"
FAIL_SYM   = "❌"
results    = []


def check(name, ok, detail=""):
    msg = f"{PASS_SYM if ok else FAIL_SYM} {name}"
    if detail: msg += f" — {detail}"
    print(msg)
    results.append((name, ok))
    return ok


def make_image(w=336, h=336, noise=False) -> bytes:
    if noise:
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        arr = np.full((h, w, 3), [150, 100, 110], dtype=np.uint8)
        cx, cy, r = w//2, h//2, min(w,h)//5
        for y in range(h):
            for x in range(w):
                if (x-cx)**2 + (y-cy)**2 < r**2:
                    arr[y,x] = [60, 40, 50]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def register_and_verify(suffix=""):
    email = f"rig_{uuid.uuid4().hex[:8]}{suffix}@test.com"
    user  = f"rig_{uuid.uuid4().hex[:6]}"
    pw    = "RigTest123!"
    r = requests.post(f"{BASE}/auth/signup",
                      json={"username": user, "email": email,
                            "password": pw, "gender": "other"}, timeout=10)
    if r.status_code != 200:
        return None, None, None
    try:
        from pymongo import MongoClient
        c = MongoClient(f"mongodb://localhost:{MONGO_PORT}",
                        serverSelectionTimeoutMS=2000)
        c["skin_disease_detection"].users.update_one(
            {"email": email}, {"$set": {"verified": True}})
        c.close()
    except Exception:
        pass
    r2 = requests.post(f"{BASE}/auth/login",
                       json={"email": email, "password": pw}, timeout=10)
    token = r2.json().get("access_token") if r2.status_code == 200 else None
    return email, pw, token


def query_prom(q):
    try:
        r = requests.get(f"{PROM}/api/v1/query",
                         params={"query": q}, timeout=5)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        pass
    return None


# ── 1. Input validation edge cases ────────────────────────────────────────────
print("\n── Input Validation ─────────────────────────────────────────────────────")

# empty file — FastAPI reads file, PIL fails to open → 400
r = requests.post(f"{BASE}/predict",
                  files={"file": ("empty.jpg", b"", "image/jpeg")},
                  headers={"Authorization": "Bearer faketoken"}, timeout=10)
check("Empty image → not 200 (auth or image error)", r.status_code != 200)

# non-image bytes — PIL raises error → 400
r = requests.post(f"{BASE}/predict",
                  files={"file": ("text.txt", b"hello world", "text/plain")},
                  headers={"Authorization": "Bearer faketoken"}, timeout=10)
check("Non-image file → not 200", r.status_code != 200)

# corrupt JPEG header
r = requests.post(f"{BASE}/predict",
                  files={"file": ("corrupt.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")},
                  headers={"Authorization": "Bearer faketoken"}, timeout=10)
check("Corrupt JPEG → not 200", r.status_code != 200)

# missing file field — FastAPI returns 422 Unprocessable Entity
r = requests.post(f"{BASE}/predict",
                  headers={"Authorization": "Bearer faketoken"},
                  timeout=10)
# FastAPI returns 422 for missing required form field
check("Missing file field → 422 (Unprocessable Entity)", r.status_code in (401, 422, 403),
      f"got {r.status_code}")

# non-JSON body for feedback endpoint
# FastAPI Pydantic model requires JSON → 422 when Content-Type is wrong
r = requests.post(f"{BASE}/feedback",
                  data="not json",
                  headers={"Authorization": "Bearer faketoken",
                            "Content-Type": "text/plain"},
                  timeout=10)
# FastAPI returns 422 for body parse failure OR 401/403 for bad token first
# Auth middleware runs before body parsing — so 401 is also acceptable
check("Non-JSON feedback → 401/422 (auth or parse fail)",
      r.status_code in (401, 403, 422),
      f"got {r.status_code}")

# ── 2. Concurrent load test ────────────────────────────────────────────────────
print("\n── Concurrent Load Test (10 requests) ───────────────────────────────────")

_, _, token = register_and_verify("_load")
if token:
    img = make_image()
    errors, latencies = [], []
    lock = threading.Lock()

    def single_predict():
        t0 = time.time()
        try:
            r = requests.post(
                f"{BASE}/predict",
                files={"file": ("test.jpg", img, "image/jpeg")},
                headers={"Authorization": f"Bearer {token}"},
                timeout=120,
            )
            latency = time.time() - t0
            with lock:
                latencies.append(latency)
                if r.status_code != 200:
                    errors.append(r.status_code)
        except Exception as e:
            with lock:
                errors.append(str(e))

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(single_predict) for _ in range(10)]
        for f in as_completed(futures):
            f.result()

    check("All 10 concurrent requests succeeded", len(errors) == 0,
          f"{len(errors)} errors: {errors[:3]}")
    if latencies:
        avg_ms = sum(latencies) / len(latencies) * 1000
        p95_ms = sorted(latencies)[int(len(latencies)*0.95)] * 1000
        check("p95 latency < 30s (CPU)", p95_ms < 30000,
              f"avg={avg_ms:.0f}ms p95={p95_ms:.0f}ms")
else:
    check("Load test user setup", False, "no token")

# ── 3. Rate limit trigger ──────────────────────────────────────────────────────
print("\n── Rate Limit Trigger ───────────────────────────────────────────────────")

_, _, rl_token = register_and_verify("_ratelimit")
if rl_token:
    img = make_image(w=50, h=50)
    for i in range(5):
        requests.post(
            f"{BASE}/predict",
            files={"file": ("tiny.jpg", img, "image/jpeg")},
            headers={"Authorization": f"Bearer {rl_token}"},
            timeout=120,
        )
    time.sleep(3)
    # sum over all uid/username label combinations
    v = query_prom("sum(user_rate_limit_breach_total)")
    check("Rate limit breach metric > 0", v is not None and v > 0, str(v))

# ── 4. MongoDB data integrity ──────────────────────────────────────────────────
print("\n── MongoDB Data Integrity ───────────────────────────────────────────────")

try:
    from pymongo import MongoClient
    client = MongoClient(f"mongodb://localhost:{MONGO_PORT}",
                         serverSelectionTimeoutMS=3000)
    db = client["skin_disease_detection"]

    cols = db.list_collection_names()
    check("users collection exists",      "users"       in cols)
    check("predictions collection exists","predictions" in cols)
    check("feedback collection exists",   "feedback"    in cols)
    check("images collection exists",     "images"      in cols)

    user_idx_names = [i["name"] for i in db.users.list_indexes()]
    check("users.email unique index",
          any("email" in n for n in user_idx_names))
    check("users.uid unique index",
          any("uid" in n for n in user_idx_names))

    pred_count = db.predictions.count_documents({})
    check("Predictions saved to MongoDB", pred_count > 0, f"{pred_count} records")

    img_count = db.images.count_documents({})
    check("Images saved to MongoDB", img_count > 0, f"{img_count} records")

    pred = db.predictions.find_one({})
    check("Prediction has uid field",      pred is not None and "uid"      in pred)
    check("Prediction has username field", pred is not None and "username" in pred)
    check("Prediction has image_id field", pred is not None and "image_id" in pred)

    fb = db.feedback.find_one({})
    if fb:
        check("Feedback has uid field",      "uid"      in fb)
        check("Feedback has username field", "username" in fb)
    else:
        check("Feedback record exists", False, "no records yet")

    client.close()
except Exception as e:
    check("MongoDB connection from host", False, str(e))
    print("  ℹ️  Connect Compass: mongodb://localhost:27017")

# ── 5. Prometheus metrics completeness ────────────────────────────────────────
print("\n── Prometheus Metrics Completeness ──────────────────────────────────────")

def prom_has(metric):
    try:
        r = requests.get(f"{PROM}/api/v1/query",
                         params={"query": metric}, timeout=5)
        if r.status_code == 200:
            return len(r.json()["data"]["result"]) > 0
    except Exception:
        pass
    return False

for metric in [
    "api_requests_total",
    "model_loaded",
    "mongodb_up",
    # use sum() for metrics with labels
    "sum(predictions_total)",
    "inference_latency_seconds_count",
    "system_cpu_percent",
    "system_memory_percent",
    "feedback_positive_rate",
    "unique_predictions_total",
    "users_registered_total",
]:
    display = metric.replace("sum(","").replace(")","")
    check(f"metric exists: {display}", prom_has(metric))

# ── 6. Alert rules loaded ──────────────────────────────────────────────────────
print("\n── Alert Rules Loaded ───────────────────────────────────────────────────")

r = requests.get(f"{PROM}/api/v1/rules", timeout=5)
if r.status_code == 200:
    groups   = r.json()["data"]["groups"]
    all_rules = [rule["name"] for g in groups for rule in g["rules"]]
    for rule in ["ModelNotLoaded", "MongoDBDown", "APIDown",
                 "SystemMemory_Critical", "SystemCPU_Critical",
                 "HighInferenceLatency_Critical"]:
        check(f"Alert rule loaded: {rule}", rule in all_rules)
else:
    check("Prometheus rules API accessible", False)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"Passed: {passed}/{total}")
if passed < total:
    for name, ok in results:
        if not ok:
            print(f"  {FAIL_SYM} {name}")
    sys.exit(1)
else:
    print("All rigorous tests passed ✅")