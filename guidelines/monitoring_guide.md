================================================================================
        DERMAI — MONITORING SETUP GUIDE (Prometheus + Grafana + AlertManager)
================================================================================

FOLDER STRUCTURE
────────────────
monitoring/
├── prometheus.yml                          ← Prometheus config
├── rules.yml                               ← Alert rules
├── alertmanager.yml                        ← AlertManager + Gmail SMTP
├── grafana_dashboard.json                  ← Dashboard (import manually)
├── grafana/
│   └── provisioning/
│       ├── datasources/prometheus.yml      ← Auto-provision datasource
│       └── dashboards/dashboard.yml        ← Auto-provision dashboard
src/
├── db/
│   ├── __init__.py
│   └── mongodb.py                          ← MongoDB client with fallback


================================================================================
1. GMAIL APP PASSWORD SETUP
================================================================================

1. Go to: myaccount.google.com → Security → 2-Step Verification (enable if not)
2. Search "App passwords" → Create new → name it "DermAI Alertmanager"
3. Copy the 16-character password (no spaces)
4. In alertmanager.yml replace:
     YOUR_GMAIL@gmail.com  → your actual gmail
     YOUR_GMAIL_APP_PASSWORD → the 16-char app password

NEVER use your real Gmail password — always use App Password.


================================================================================
2. WITHOUT DOCKER — MANUAL SETUP
================================================================================

── Prometheus ──────────────────────────────────────────────────────────────────

Download: https://prometheus.io/download/

    # extract and run
    ./prometheus --config.file=monitoring/prometheus.yml \
                 --web.listen-address=:9090 \
                 --storage.tsdb.path=./prometheus_data

    # for shared machine — use a free port
    ./prometheus --config.file=monitoring/prometheus.yml \
                 --web.listen-address=:9091   # change if 9090 taken

Open: http://localhost:9090

── AlertManager ────────────────────────────────────────────────────────────────

Download: https://prometheus.io/download/#alertmanager

    ./alertmanager --config.file=monitoring/alertmanager.yml \
                   --web.listen-address=:9093

── Grafana ─────────────────────────────────────────────────────────────────────

Download: https://grafana.com/grafana/download

    # Ubuntu/Debian
    sudo apt install grafana
    sudo systemctl start grafana-server

    # or run binary directly
    ./bin/grafana-server --homepath=/usr/share/grafana \
                         --config=/etc/grafana/grafana.ini \
                         web

Open: http://localhost:3000  (default admin/admin)

── MongoDB ─────────────────────────────────────────────────────────────────────

    # Ubuntu
    sudo apt install mongodb
    sudo systemctl start mongodb

    # or via pip (lightweight alternative for dev)
    pip install pymongo


================================================================================
3. GRAFANA CONFIGURATION
================================================================================

Step 1 — Add Prometheus datasource:
    Configuration → Data Sources → Add → Prometheus
    URL: http://localhost:9090
    Save & Test

Step 2 — Import Dashboard:
    Dashboards → Import → Upload JSON file
    Select: monitoring/grafana_dashboard.json
    Select datasource: Prometheus
    Import

Step 3 — Configure Alerting (Grafana Alerts):
    Alerting → Contact Points → Add → Email
    Address: your@gmail.com
    Test → Save

Step 4 — Set dashboard refresh:
    Top right of dashboard → set to "10s" or "30s"


================================================================================
4. VERIFY EVERYTHING IS WORKING
================================================================================

    # check FastAPI is exposing metrics
    curl http://localhost:8000/metrics | grep api_requests_total

    # check Prometheus is scraping
    # open http://localhost:9090 → Status → Targets
    # fastapi_backend should show "UP" in green

    # send a test prediction
    curl -X POST http://localhost:8000/predict \
         -F "file=@path/to/skin_image.jpg"

    # check metric appeared in Prometheus
    # query: predictions_total
    # should show count per class

    # check alert rules loaded
    # http://localhost:9090 → Alerts
    # should show all rules from rules.yml


================================================================================
5. PROMETHEUS QUERIES FOR GRAFANA PANELS
================================================================================

Inference Latency p95:
    histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))

Request Rate:
    rate(api_requests_total[5m])

Error Rate:
    rate(errors_total[5m])

Status Code Pie:
    sum by (status_code) (api_requests_total)

Predictions by Class:
    sum by (predicted_class) (predictions_total)

Like/Dislike Ratio:
    sum by (vote) (feedback_total)

CPU Usage:
    system_cpu_percent

Memory Usage:
    system_memory_percent

GPU Memory:
    gpu_memory_allocated_gb

MongoDB Status:
    mongodb_up

Active Requests:
    sum(active_requests)

Confidence Median:
    histogram_quantile(0.50, rate(prediction_confidence_bucket[10m]))


================================================================================
6. ALERT RULES SUMMARY
================================================================================

CRITICAL (immediate email, repeat every 30m):
    HighInferenceLatency_Critical   p95 latency > 3s for 2m
    HighErrorRate_Critical          error rate > 0.1/s for 2m
    ModelNotLoaded                  model_loaded == 0 for 1m
    SystemMemory_Critical           memory > 90% for 2m
    GPUMemory_Critical              GPU alloc > 40GB for 2m
    SystemCPU_Critical              CPU > 90% for 3m
    APIDown                         backend unreachable for 1m
    MongoDBDown                     mongodb_up == 0 for 1m
    MongoDBDown_Sustained           mongodb_up == 0 for 5m

WARNING (batched email, repeat every 2h):
    HighInferenceLatency_Moderate   p95 latency > 1s for 5m
    HighRiskPredictionSpike         melanoma rate > 0.5/s for 5m
    LowPredictionConfidence         median confidence < 40% for 10m
    SystemMemory_Moderate           memory > 75% for 5m
    GPUMemory_Moderate              GPU alloc > 30GB for 5m
    SystemCPU_Moderate              CPU > 70% for 5m
    RequestCPUSpike                 p95 per-req CPU > 60% for 5m
    RequestMemorySpike              p95 per-req memory > 1GB for 3m
    HighActiveRequests              concurrent requests > 20 for 2m
    NegativeFeedbackSpike           thumbs_down 2x thumbs_up for 10m


================================================================================
7. PORT ASSIGNMENTS (shared machine safe)
================================================================================

Service              Port    ENV variable
──────────────────────────────────────────
FastAPI backend      8000    API_PORT
Frontend             7500    FRONTEND_PORT
MLflow server        5001    MLFLOW_PORT
Prometheus           9090    PROMETHEUS_PORT
Grafana              3000    GRAFANA_PORT
AlertManager         9093    ALERTMANAGER_PORT
MongoDB              27017   MONGODB_PORT
Node Exporter        9182    NODE_EXPORTER_PORT

If any port is taken on shared machine, change in .env and update prometheus.yml targets.

Check free ports:
    for port in 8000 7500 5001 9090 3000 9093 27017; do
        lsof -i :$port > /dev/null 2>&1 && echo "$port IN USE" || echo "$port FREE"
    done


================================================================================
8. MONGODB USAGE IN API
================================================================================

The mongodb.py module has automatic fallback:
- If MongoDB is reachable → stores in MongoDB
- If MongoDB is down → stores in memory dict (temporary)

To install pymongo:
    pip install pymongo

MongoDB collections:
    feedback    — user thumbs up/down per prediction
    predictions — all predictions with class, confidence, timestamp

View data:
    mongosh
    use skin_disease_detection
    db.feedback.find().pretty()
    db.predictions.find().sort({timestamp:-1}).limit(10).pretty()


================================================================================