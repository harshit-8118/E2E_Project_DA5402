#!/usr/bin/env python3
"""
tests/test_easy.py
Easy-level tests — basic connectivity and health checks
Run: python tests/test_easy.py
"""

import sys
import json
import time
import requests

BASE = "http://localhost:8000"
FRONTEND = "http://localhost:7500"
PROMETHEUS = "http://localhost:9090"
GRAFANA = "http://localhost:3005"

PASS = "✅"
FAIL = "❌"
results = []


def check(name, ok, detail=""):
    status = PASS if ok else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, ok))
    return ok


def get(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        return r
    except Exception as e:
        return None


# ── 1. Service health ──────────────────────────────────────────────────────────
print("\n── Service Health ──────────────────────────────────────────────────────")

r = get(f"{BASE}/health")
check("Backend /health responds", r is not None and r.status_code == 200,
      r.json().get("status") if r else "unreachable")

r = get(f"{BASE}/ready")
check("Backend /ready — model loaded", r is not None and r.status_code == 200,
      r.json().get("status") if r else "model not loaded")

r = get(FRONTEND)
check("Frontend serves HTML", r is not None and r.status_code == 200)

r = get(f"{PROMETHEUS}/-/healthy")
check("Prometheus healthy", r is not None and r.status_code == 200)

r = get(f"{GRAFANA}/api/health", timeout=5)
check("Grafana healthy", r is not None and r.status_code == 200)

# ── 2. API endpoints exist ─────────────────────────────────────────────────────
print("\n── API Endpoints ────────────────────────────────────────────────────────")

r = get(f"{BASE}/classes")
check("/classes returns 7 classes",
      r is not None and len(r.json().get("classes", [])) == 7)

r = get(f"{BASE}/system/info")
check("/system/info responds", r is not None and r.status_code == 200,
      f"device={r.json().get('device','?')}" if r else "")

r = get(f"{BASE}/metrics")
check("/metrics endpoint returns Prometheus text",
      r is not None and "api_requests_total" in r.text)

# ── 3. Auth endpoints exist ────────────────────────────────────────────────────
print("\n── Auth Endpoints ───────────────────────────────────────────────────────")

r = requests.post(f"{BASE}/auth/login",
                  json={"email": "nonexistent@test.com", "password": "wrong"},
                  timeout=5)
check("/auth/login returns 401 for bad credentials", r.status_code == 401)

r = requests.post(f"{BASE}/auth/signup",
                  json={"username": "x", "email": "bad", "password": "short"},
                  timeout=5)
check("/auth/signup rejects invalid input", r.status_code in (422, 409))

# ── 4. Prometheus scraping ─────────────────────────────────────────────────────
print("\n── Prometheus Targets ───────────────────────────────────────────────────")

r = get(f"{PROMETHEUS}/api/v1/targets")
if r and r.status_code == 200:
    targets = r.json()["data"]["activeTargets"]
    up_jobs = [t["labels"]["job"] for t in targets if t["health"] == "up"]
    check("fastapi_backend target UP", "fastapi_backend" in up_jobs)
    check("prometheus target UP", "prometheus" in up_jobs)
    check("node_exporter target UP", "node_exporter" in up_jobs)
else:
    check("Prometheus targets API", False, "unreachable")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"Passed: {passed}/{total}")
if passed < total:
    print("Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"  {FAIL} {name}")
    sys.exit(1)
else:
    print("All easy tests passed ✅")