#!/bin/bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../../
PATH_CONTAINING_PROJECT="$(pwd)"

cd UltraFlwr

# Define the HOME directory for result storage
HOME=$(pwd)

# Create logs directory if it doesn't exist
LOG_DIR="$HOME/logs/test_logs"
mkdir -p "$LOG_DIR"

# Check if we can write to the logs directory
if [[ ! -w "$LOG_DIR" ]]; then
    echo "Warning: Cannot write to $LOG_DIR. Trying to fix permissions..."
    chmod 755 "$HOME/logs" 2>/dev/null || true
    chmod 755 "$LOG_DIR" 2>/dev/null || true
fi

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/test_run_${TIMESTAMP}.log"

# Test if we can create the log file
if ! touch "$MAIN_LOG" 2>/dev/null; then
    echo "Warning: Cannot create log file at $MAIN_LOG. Using /tmp for logging..."
    LOG_DIR="/tmp/test_logs"
    mkdir -p "$LOG_DIR"
    MAIN_LOG="$LOG_DIR/test_run_${TIMESTAMP}.log"
fi

# Start logging
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "Test run started at $(date)"
echo "Logs will be saved to: $MAIN_LOG"
echo "========================================"

PYTHON_SCRIPT="FedYOLO/test/test.py"

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# Install FedYOLO from pyproject.toml, uncomment if already installed
if [[ -f "pyproject.toml" ]]; then
    echo "Installing FedYOLO package..."
    pip install --no-cache-dir -e . > /dev/null
else
    echo "Error: pyproject.toml not found. Cannot install FedYOLO."
    exit 1
fi

sed -i "s|^BASE = .*|BASE = \"$PATH_CONTAINING_PROJECT\"|" "$CLIENT_CONFIG_FILE"

# List of datasets and strategies (similar to benchmark.sh)
# DATASET_NAME_LIST=("mnist")
# STRATEGY_LIST=("FedAvg" "FedHeadAvg" "FedHeadMedian" "FedNeckAvg" "FedNeckMedian" "FedBackboneAvg" "FedBackboneMedian" "FedNeckHeadAvg" "FedNeckHeadMedian")
STRATEGY_LIST=(
    "FedAvg"
    "FedHeadAvg"
)

# Number of clients for client-dependent tests
NUM_CLIENTS=$(python3 -c "from FedYOLO.config import NUM_CLIENTS; print(NUM_CLIENTS)")
echo "Number of clients: $NUM_CLIENTS"

# Read CLIENT_CONFIG as JSON
CLIENT_CONFIG=$(python3 -c "
import sys
sys.path.append('./FedYOLO')
from config import CLIENT_CONFIG
import json
print(json.dumps(CLIENT_CONFIG))
")

# Parse CLIENT_CONFIG using jq
echo "Client Configuration:"
echo "$CLIENT_CONFIG" | jq

# Define scoring styles
CLIENT_DEPENDENT_STYLES=("client-client" "client-server" "server-client")
CLIENT_INDEPENDENT_STYLES=("server-server")

# Initialize a variable to store the first client's task
FIRST_TASK=$(echo "$CLIENT_CONFIG" | jq -r '."0".task')
ALL_TASKS_SAME=true  # Assume all tasks are the same initially

for ((CLIENT_ID=0; CLIENT_ID<NUM_CLIENTS; CLIENT_ID++)); do
    CLIENT_TASK=$(echo "$CLIENT_CONFIG" | jq -r ".\"$CLIENT_ID\".task")

    # Compare the current client's task with the first client's task
    if [[ "$CLIENT_TASK" != "$FIRST_TASK" ]]; then
        ALL_TASKS_SAME=false
        break
    fi
done

# Output the result
if [[ "$ALL_TASKS_SAME" == true ]]; then
    echo "All clients have the same task: $FIRST_TASK"
else
    echo "Clients have different tasks."
fi

# ...existing code...

# Initialize an array to store test details
declare -a TEST_SUMMARY

# Loops over strategies
for STRATEGY in "${STRATEGY_LIST[@]}"; do
    echo "===================================================================="
    echo "Running tests for STRATEGY=${STRATEGY}"
    echo "===================================================================="

    IS_PARTIAL_AGGREGATION=false

    # Check if STRATEGY contains head, neck, or backbone
    if [[ "$STRATEGY" == *"Head"* || "$STRATEGY" == *"Neck"* || "$STRATEGY" == *"Backbone"* ]]; then
        IS_PARTIAL_AGGREGATION=true
    fi

    # Loops over clients
    for ((CLIENT_ID=0; CLIENT_ID<NUM_CLIENTS; CLIENT_ID++)); do
        CLIENT_DATA=$(echo "$CLIENT_CONFIG" | jq ".\"$CLIENT_ID\"")  # Access key as a string
        CLIENT_TASK=$(echo "$CLIENT_DATA" | jq -r '.task')
        CLIENT_DATASET=$(echo "$CLIENT_DATA" | jq -r '.data_path')
        CLIENT_DATASET_NAME=$(echo "$CLIENT_DATA" | jq -r '.dataset_name')

        echo "Processing Client $CLIENT_ID"
        echo "  Task: $CLIENT_TASK"
        echo "  Dataset Path: $CLIENT_DATASET"
        echo "  Dataset Name: $CLIENT_DATASET_NAME"

        # Evaluate the client on its own data
        echo "Evaluating Client $CLIENT_ID on its own data"
        TEST_LOG="$LOG_DIR/client_${CLIENT_ID}_own_data_${STRATEGY}_${TIMESTAMP}.log"
        python3 "$PYTHON_SCRIPT" --dataset_name "$CLIENT_DATASET_NAME" --strategy_name "$STRATEGY" --client_num "$CLIENT_ID" --scoring_style "client-client" --task "$CLIENT_TASK" --data_path "$CLIENT_DATASET" > "$TEST_LOG" 2>&1
        TEST_SUMMARY+=("Client $CLIENT_ID evaluated on its own data with STRATEGY=$STRATEGY - Log: $TEST_LOG")

        if [[ "$ALL_TASKS_SAME" == true ]]; then
            # Evaluate the client on data from all other clients
            for ((OTHER_CLIENT_ID=0; OTHER_CLIENT_ID<NUM_CLIENTS; OTHER_CLIENT_ID++)); do
                if [[ "$CLIENT_ID" -ne "$OTHER_CLIENT_ID" ]]; then
                    OTHER_CLIENT_DATA=$(echo "$CLIENT_CONFIG" | jq ".\"$OTHER_CLIENT_ID\"")
                    OTHER_CLIENT_DATASET_NAME=$(echo "$OTHER_CLIENT_DATA" | jq -r '.dataset_name')
                    OTHER_CLIENT_DATASET=$(echo "$OTHER_CLIENT_DATA" | jq -r '.data_path')

                    echo "Evaluating Client $CLIENT_ID on data from Client $OTHER_CLIENT_ID..."
                    TEST_LOG="$LOG_DIR/client_${CLIENT_ID}_on_client_${OTHER_CLIENT_ID}_data_${STRATEGY}_${TIMESTAMP}.log"
                    python3 "$PYTHON_SCRIPT" --dataset_name "$OTHER_CLIENT_DATASET_NAME" --strategy_name "$STRATEGY" --client_num "$CLIENT_ID" --scoring_style "client-client" --task "$CLIENT_TASK" --data_path "$OTHER_CLIENT_DATASET" --data_source_client "$OTHER_CLIENT_ID" > "$TEST_LOG" 2>&1
                    TEST_SUMMARY+=("Client $CLIENT_ID evaluated on data from Client $OTHER_CLIENT_ID with STRATEGY=$STRATEGY - Log: $TEST_LOG")
                fi
            done

            # Evaluate the client on data from the server
            echo "Evaluating Client $CLIENT_ID on data from the server..."
            TEST_LOG="$LOG_DIR/client_${CLIENT_ID}_on_server_data_${STRATEGY}_${TIMESTAMP}.log"
            python3 "$PYTHON_SCRIPT" --dataset_name "$CLIENT_DATASET_NAME" --strategy_name "$STRATEGY" --client_num "$CLIENT_ID" --scoring_style "client-server" --task "$CLIENT_TASK" > "$TEST_LOG" 2>&1
            TEST_SUMMARY+=("Client $CLIENT_ID evaluated on data from the server with STRATEGY=$STRATEGY - Log: $TEST_LOG")
        fi
    done

    # Server model evaluation (only for full aggregation strategies)
    if [[ "$IS_PARTIAL_AGGREGATION" == false && "$ALL_TASKS_SAME" == true ]]; then
        # Use the first client's dataset info for server evaluation
        FIRST_CLIENT_DATA=$(echo "$CLIENT_CONFIG" | jq ".\"0\"")
        FIRST_CLIENT_DATASET_NAME=$(echo "$FIRST_CLIENT_DATA" | jq -r '.dataset_name')
        FIRST_CLIENT_TASK=$(echo "$FIRST_CLIENT_DATA" | jq -r '.task')

        # Evaluate the server model on data from the server
        echo "Evaluating server model on data from the server..."
        TEST_LOG="$LOG_DIR/server_on_server_data_${STRATEGY}_${TIMESTAMP}.log"
        python3 "$PYTHON_SCRIPT" --dataset_name "$FIRST_CLIENT_DATASET_NAME" --strategy_name "$STRATEGY" --scoring_style "server-server" --task "$FIRST_CLIENT_TASK" --client_num 0 > "$TEST_LOG" 2>&1
        TEST_SUMMARY+=("Server model evaluated on data from the server with STRATEGY=$STRATEGY - Log: $TEST_LOG")

        # Evaluate the server model on data from all clients
        for ((CLIENT_ID=0; CLIENT_ID<NUM_CLIENTS; CLIENT_ID++)); do
            CLIENT_DATA=$(echo "$CLIENT_CONFIG" | jq ".\"$CLIENT_ID\"")
            CLIENT_DATASET_NAME=$(echo "$CLIENT_DATA" | jq -r '.dataset_name')
            CLIENT_DATASET=$(echo "$CLIENT_DATA" | jq -r '.data_path')

            echo "Evaluating server model on data from Client $CLIENT_ID..."
            TEST_LOG="$LOG_DIR/server_on_client_${CLIENT_ID}_data_${STRATEGY}_${TIMESTAMP}.log"
            python3 "$PYTHON_SCRIPT" --dataset_name "$CLIENT_DATASET_NAME" --strategy_name "$STRATEGY" --scoring_style "server-client" --task "$FIRST_CLIENT_TASK" --data_path "$CLIENT_DATASET" --client_num "$CLIENT_ID" > "$TEST_LOG" 2>&1
            TEST_SUMMARY+=("Server model evaluated on data from Client $CLIENT_ID with STRATEGY=$STRATEGY - Log: $TEST_LOG")
        done
    fi
done

# Print the summary of all tests performed
echo "===================================================================="
echo "Summary of Tests Performed:"
echo "===================================================================="
for TEST in "${TEST_SUMMARY[@]}"; do
    echo "$TEST"
done
