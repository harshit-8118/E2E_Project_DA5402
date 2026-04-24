# run in bash
# ./scripts/check_port.sh

PORTS=(7500 8000 27017 5000 9090 9093 3000 9182 8080 3005)
for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "PORT $port — IN USE"
    else
        echo "PORT $port — FREE"
    fi
done