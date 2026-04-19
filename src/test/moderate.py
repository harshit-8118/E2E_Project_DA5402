#!/usr/bin/env python3
"""
tests/test_moderate.py
Moderate-level tests — full auth flow, prediction, feedback, metrics validation
Run: python tests/test_moderate.py
"""

import sys
import io
import time
import uuid
import requests
import numpy as np
from PIL import Image

BASE    = "http://localhost:8000"
PROM    = "http://localhost:9090"
PASS_SYM = "✅"
FAIL_SYM = "❌"
results  = []
_token   = None
_pred_id = None


def check(name, ok, detail=""):
    msg = f"{PASS_SYM if ok else FAIL_SYM} {name}"
    if detail: msg += f" — {detail}"
    print(msg)
    results.append((name, ok))
    return ok


def make_test_image(w=336, h=336) -> bytes:
    arr = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 4
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                arr[y, x] = [80, 50, 60]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def auth_headers():
    return {"Authorization": f"Bearer {_token}"}


def query_prom(q):
    """Query Prometheus and return float value or None."""
    try:
        r = requests.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=5)
        if r.status_code == 200:
            result = r.json()["data"]["result"]
            return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        pass
    return None


# ── 1. Registration ────────────────────────────────────────────────────────────
print("\n── User Registration ─────────────────────────────────────────────────────")

test_email = f"test_{uuid.uuid4().hex[:8]}@dermai-test.com"
test_user  = f"testuser_{uuid.uuid4().hex[:6]}"
test_pass  = "TestPass123!"

r = requests.post(f"{BASE}/auth/signup", json={
    "username": test_user, "email": test_email,
    "password": test_pass, "gender": "prefer_not_to_say",
}, timeout=10)
check("Signup returns 200", r.status_code == 200, r.text[:80])
uid = r.json().get("uid") if r.status_code == 200 else None
check("Signup returns uid", uid is not None)

r2 = requests.post(f"{BASE}/auth/signup", json={
    "username": test_user, "email": f"other_{test_email}",
    "password": test_pass, "gender": "male",
}, timeout=10)
check("Duplicate username rejected (409)", r2.status_code == 409)

# ── 2. OTP + Login ─────────────────────────────────────────────────────────────
print("\n── OTP + Login ──────────────────────────────────────────────────────────")

mongo_direct = False
try:
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
    db = client["skin_disease_detection"]
    db.users.update_one({"email": test_email}, {"$set": {"verified": True}})
    client.close()
    check("MongoDB: force-verified test user", True)
    mongo_direct = True
except Exception as e:
    check("MongoDB direct access (optional)", False, str(e))

r = requests.post(f"{BASE}/auth/login",
                  json={"email": test_email, "password": test_pass}, timeout=10)
if mongo_direct:
    check("Login returns 200", r.status_code == 200, r.text[:80])
    _token = r.json().get("access_token") if r.status_code == 200 else None
    check("Login returns JWT token", _token is not None)
else:
    print("  ⚠️  Skipping login — MongoDB not directly accessible from host")

# ── 3. Prediction ──────────────────────────────────────────────────────────────
print("\n── Prediction ───────────────────────────────────────────────────────────")

if _token:
    img_bytes = make_test_image()
    r = requests.post(
        f"{BASE}/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        headers=auth_headers(),
        timeout=120,
    )
    check("POST /predict returns 200", r.status_code == 200, r.text[:100])

    if r.status_code == 200:
        data = r.json()
        _pred_id = data.get("prediction_id")
        check("Response has prediction_id", _pred_id is not None)
        check("predicted_class is valid",
              data.get("predicted_class") in ["akiec","bcc","bkl","df","mel","nv","vasc"])
        check("confidence in (0, 1]", 0 < data.get("confidence", 0) <= 1)
        check("all_scores has 7 classes", len(data.get("all_scores", {})) == 7)
        check("symptoms list non-empty", len(data.get("symptoms", [])) > 0)
        check("advisory non-empty", len(data.get("advisory", "")) > 0)
        check("disclaimer present", "NOT" in data.get("disclaimer", ""))
        check("gradcam_image is base64 PNG",
              data.get("gradcam_image", "").startswith("data:image/png;base64,"))
        check("image_saved flag present", "image_saved" in data)

    r_unauth = requests.post(
        f"{BASE}/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=10,
    )
    check("Unauthenticated /predict returns 401/403",
          r_unauth.status_code in (401, 403))
else:
    print("  ⚠️  Skipping predict tests — no token")

# ── 4. Feedback ────────────────────────────────────────────────────────────────
print("\n── Feedback ─────────────────────────────────────────────────────────────")

if _token and _pred_id:
    r = requests.post(f"{BASE}/feedback",
                      json={"prediction_id": _pred_id, "vote": "thumbs_up",
                            "comment": "test feedback"},
                      headers=auth_headers(), timeout=10)
    check("POST /feedback thumbs_up", r.status_code == 200)
    check("Feedback stored_in field present",
          "stored_in" in (r.json() if r.status_code == 200 else {}))

    r = requests.post(f"{BASE}/feedback",
                      json={"prediction_id": _pred_id, "vote": "invalid"},
                      headers=auth_headers(), timeout=10)
    check("Invalid vote rejected (422)", r.status_code == 422)

    r = requests.get(f"{BASE}/feedback/stats", headers=auth_headers(), timeout=10)
    check("/feedback/stats returns total > 0", r.json().get("total", 0) > 0)

# ── 5. User profile ────────────────────────────────────────────────────────────
print("\n── User Profile ─────────────────────────────────────────────────────────")

if _token:
    r = requests.get(f"{BASE}/auth/me", headers=auth_headers(), timeout=10)
    check("GET /auth/me returns 200", r.status_code == 200)
    if r.status_code == 200:
        me = r.json()
        check("Profile has prediction_count", "prediction_count" in me)
        check("prediction_count >= 1", me.get("prediction_count", 0) >= 1)

    r = requests.get(f"{BASE}/auth/my-predictions", headers=auth_headers(), timeout=10)
    check("GET /auth/my-predictions returns list", r.status_code == 200)

# ── 6. Prometheus metrics ──────────────────────────────────────────────────────
print("\n── Prometheus Metrics ───────────────────────────────────────────────────")

time.sleep(2)   # allow prometheus scrape to catch up

v = query_prom("api_requests_total")
check("api_requests_total > 0", v is not None and v > 0, str(v))

v = query_prom("model_loaded")
check("model_loaded == 1", v == 1.0, str(v))

v = query_prom("mongodb_up")
check("mongodb_up == 1", v == 1.0, str(v))

if _token:
    # CORRECT query: sum over all label combinations
    v = query_prom("sum(predictions_total)")
    check("sum(predictions_total) > 0", v is not None and v > 0, str(v))

    v = query_prom("sum(user_predictions_total)")
    check("sum(user_predictions_total) > 0", v is not None and v > 0, str(v))

    v = query_prom("unique_predictions_total")
    check("unique_predictions_total > 0", v is not None and v > 0, str(v))

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
    print("All moderate tests passed ✅")