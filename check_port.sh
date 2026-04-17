PORTS=(8000 3000 5001 9090 3001 9093 27017)
for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "PORT $port — IN USE"
    else
        echo "PORT $port — FREE"
    fi
done