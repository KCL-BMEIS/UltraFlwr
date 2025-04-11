#!/bin/bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH
cd ../../

PYTHON_SCRIPT="FedYOLO/test/test.py"
CONFIG_FILE="FedYOLO/train/yolo_client.py"

# Install FedYOLO from pyproject.toml, uncomment if already installed
if [[ -f "pyproject.toml" ]]; then
    echo "Installing FedYOLO package..."
    pip install --no-cache-dir -e .
else
    echo "Error: pyproject.toml not found. Cannot install FedYOLO."
    exit 1
fi

# List of datasets and strategies (similar to benchmark.sh)
DATASET_NAME_LIST=("baseline")
# STRATEGY_LIST=("FedAvg" "FedHeadAvg" "FedHeadMedian" "FedNeckAvg" "FedNeckMedian" "FedBackboneAvg" "FedBackboneMedian" "FedNeckHeadAvg" "FedNeckHeadMedian")
STRATEGY_LIST=("FedNeckMedian" "FedBackboneAvg")

# Number of clients for client-dependent tests
NUM_CLIENTS=$(python3 -c "from FedYOLO.config import NUM_CLIENTS; print(NUM_CLIENTS)")

# Define scoring styles
CLIENT_DEPENDENT_STYLES=("client-client" "client-server" "server-client")
CLIENT_INDEPENDENT_STYLES=("server-server")

# Function to check if strategy contains head, neck, or backbone
should_skip_server() {
    local strategy=$1
    if [[ "$strategy" == *"Head"* || "$strategy" == *"Neck"* || "$strategy" == *"Backbone"* ]]; then
        return 0  # true (skip server-server or server-client tests)
    else
        return 1  # false (run all tests)
    fi
}

# Loop over datasets and strategies
for DATASET_NAME in "${DATASET_NAME_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        echo "===================================================================="
        echo "Running tests for DATASET_NAME=${DATASET_NAME}, STRATEGY=${STRATEGY}"
        echo "===================================================================="

        # Modify config.py file to set the current dataset and strategy
        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" "$CONFIG_FILE"
        sed -i "s/^\s*'strategy': .*/    'strategy': '${STRATEGY}',/" "$CONFIG_FILE"

        # Run client-independent (server-server) tests
        if ! should_skip_server "$STRATEGY"; then
            for SCORING_STYLE in "${CLIENT_INDEPENDENT_STYLES[@]}"; do
                echo "Running client-independent test: scoring_style=${SCORING_STYLE}"
                python3 "$PYTHON_SCRIPT" --dataset_name "$DATASET_NAME" --strategy_name "$STRATEGY" --scoring_style "$SCORING_STYLE"
                echo ""
            done
        else
            echo "Skipping server-based tests for STRATEGY=${STRATEGY} (contains head/neck/backbone)"
            echo ""
        fi

        # Run client-dependent tests (client-client, client-server)
        for ((CLIENT_NUM=0; CLIENT_NUM<NUM_CLIENTS; CLIENT_NUM++)); do  # Simulating client IDs (adjust as needed)
            for SCORING_STYLE in "${CLIENT_DEPENDENT_STYLES[@]}"; do
                if should_skip_server "$STRATEGY" && [[ "$SCORING_STYLE" == "server-client" ]]; then
                    echo "Skipping server-client test for STRATEGY=${STRATEGY} (contains head/neck/backbone)"
                    echo ""
                    continue
                fi

                echo "Running client-dependent test: client_num=${CLIENT_NUM}, scoring_style=${SCORING_STYLE}"
                python3 "$PYTHON_SCRIPT" --dataset_name "$DATASET_NAME" --strategy_name "$STRATEGY" --client_num "$CLIENT_NUM" --scoring_style "$SCORING_STYLE"
                echo ""
            done
        done
    done
done
