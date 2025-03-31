#!/usr/bin/env bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

# Default values for arguments
SERVER_SCRIPT="FedYOLO/train/yolo_server.py"
CLIENT_SCRIPT="FedYOLO/train/yolo_client.py"
SERVER_ADDRESS="127.0.0.1:8080"  # Changed to localhost for better connectivity

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

DATASET_NAME=$(python3 -c "from FedYOLO.config import SPLITS_CONFIG; print(SPLITS_CONFIG['dataset_name'])")
STRATEGY_NAME=$(python3 -c "from FedYOLO.config import SERVER_CONFIG; print(SERVER_CONFIG['strategy'])")

# Start superlink
start_superlink () {
    # Free port 9092 before starting the server
    echo "Freeing ports 9091, 9092, 9093..."
    lsof -t -i:9091 -i:9092 -i:9093 | xargs kill -9 2>/dev/null
    flower-superlink --insecure 2>/dev/null &
    SERVER_PID=$!
    PIDS+=($SERVER_PID)
    echo "Server started with PID: $SERVER_PID."
}

start_app () {
    SERVER_LOG="logs/server_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
    PYTHONUNBUFFERED=1 flwr run . local-deployment --stream > "$SERVER_LOG" 2>&1 &

    APP_PID=$!
    PIDS+=($APP_PID)

    echo "Server started with PID: $APP_PID"

    # Monitor the log for the word "finished"
    tail -F "$SERVER_LOG" | while read line; do
        echo "$line" | grep -q "finished"
        if [[ $? -eq 0 ]]; then
            echo "✅ Detected 'finished' in server log. Exiting..."
            sleep 2
            kill "${PIDS[@]}" 2>/dev/null
            exit 0
        fi
    done
}

# Function to start a supernode
start_supernode() {
    CLIENT_CID=$1
    CLIENT_DATA_PATH=$(python3 -c "from FedYOLO.config import CLIENT_CONFIG; print(CLIENT_CONFIG[$CLIENT_CID]['data_path'])")
    CLIENT_LOG="logs/client_${CLIENT_CID}_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
    
    # Dynamically compute port as 909(A + 4)
    PORT=$((9090 + CLIENT_CID + 4))

    echo "Freeing port ${PORT}..."
    lsof -t -i:${PORT} | xargs kill -9 2>/dev/null

    echo "Starting supernode for client $CLIENT_CID with data path: $CLIENT_DATA_PATH..."
    echo "ClientAppIO API address: 127.0.0.1:${PORT}"

    flower-supernode \
      --insecure \
      --superlink 127.0.0.1:9092 \
      --clientappio-api-address 127.0.0.1:${PORT} \
      --node-config "cid=${CLIENT_CID} data_path=\"${CLIENT_DATA_PATH}\"" > "$CLIENT_LOG" 2>&1 &

    CLIENT_PID=$!
    PIDS+=($CLIENT_PID)
    echo "Client $CLIENT_CID started with PID: $CLIENT_PID. Logs: $CLIENT_LOG"
}


# # Function to start the server
# start_server() {
#     # Free port 8080 before starting the server
#     echo "Freeing port 8080..."
#     lsof -t -i:8080 | xargs kill -9 2>/dev/null
#     echo "Starting server..."
#     SERVER_LOG="logs/server_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
#     python3 "$SERVER_SCRIPT" > "$SERVER_LOG" 2>&1 &
#     SERVER_PID=$!
#     PIDS+=($SERVER_PID)
#     echo "Server started with PID: $SERVER_PID. Logs: $SERVER_LOG"
# }

# Function to start a client with its own config
# start_client() {
#     CLIENT_CID=$1
#     CLIENT_DATA_PATH=$(python3 -c "from FedYOLO.config import CLIENT_CONFIG; print(CLIENT_CONFIG[$CLIENT_CID]['data_path'])")
#     CLIENT_LOG="logs/client_${CLIENT_CID}_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
#     echo "Starting client $CLIENT_CID with data path: $CLIENT_DATA_PATH..."
#     python3 "$CLIENT_SCRIPT" --cid="$CLIENT_CID" --data_path="$CLIENT_DATA_PATH" > "$CLIENT_LOG" 2>&1 &
#     CLIENT_PID=$!
#     PIDS+=($CLIENT_PID)
#     echo "Client $CLIENT_CID started with PID: $CLIENT_PID. Logs: $CLIENT_LOG"
# }

# Start the server
start_superlink

# Add a short delay to ensure server is up
sleep 2

# Start clients based on CLIENT_CONFIG
for CLIENT_CID in $(python3 -c "from FedYOLO.config import CLIENT_CONFIG; print(' '.join(map(str, CLIENT_CONFIG.keys())))"); do
    start_supernode "$CLIENT_CID"
done

start_app

# Wait for all processes to finish
wait
