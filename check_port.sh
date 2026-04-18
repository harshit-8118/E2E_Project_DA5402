# run in bash
# API_PORT=8000          # FastAPI backend
# FRONTEND_PORT=7500     # frontend HTML server
# MLFLOW_PORT=5001       # mlflow server (serve.sh)
# PROMETHEUS_PORT=9090   # prometheus
# GRAFANA_PORT=3000      # grafana
# ALERTMANAGER_PORT=9093 # alertmanager
# NODE_EXPORTER=9182     # node_exporter
# MONGODB_PORT=27017     # mongodb

PORTS=(7500 8000 27017 5001 9090 9093 3000 9182)
for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "PORT $port — IN USE"
    else
        echo "PORT $port — FREE"
    fi
done