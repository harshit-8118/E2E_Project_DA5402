#!/bin/bash
set -euo pipefail

# 1. Configuration
# UPDATED: Point to the 'production' alias of the registered model
# PRODUCTION MODLE RUNID=f8a0be4f4bb34682baaef1f809f5de4b
MODEL_URI="${1:-models:/skin-disease-classifier@production}"
PORT="${PORT:-5001}"

# 2. Load .env properly
if [ -f .env ]; then
    echo "--- Loading environment variables from .env ---"
    set -a
    source .env
    set +a
fi

# Ensure tracking URI is set to DagsHub
export MLFLOW_TRACKING_URI="https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MLflow Model Server (DagsHub Remote)"
echo "  Model URI : $MODEL_URI"
echo "  Local Port: $PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 3. Start the Server 
echo "Downloading model from Registry and starting server..."
mlflow models serve \
    -m "$MODEL_URI" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --env-manager local