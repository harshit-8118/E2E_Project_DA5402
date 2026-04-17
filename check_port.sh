PORTS=(8000 3000 6000 6300 5000 8080 5001 5002 9090 3001 9093 27017)
for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "PORT $port — IN USE"
    else
        echo "PORT $port — FREE"
    fi
done