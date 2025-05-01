#!/bin/bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../../
PATH_CONTAINING_PROJECT="$(pwd)"

cd UltraFlwr

PYTHON_SCRIPT="FedYOLO/test/test.py"
CONFIG_FILE="FedYOLO/train/yolo_client.py"

# Define the HOME directory for result storage
HOME=$(pwd)

# List of datasets and strategies (similar to benchmark.sh)
DATASET_NAME_LIST=("surg_od")
STRATEGY_LIST=("FedNeckMedian")

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# Install FedYOLO from pyproject.toml, uncomment if already installed
if [[ -f "pyproject.toml" ]]; then
    echo "Installing FedYOLO package..."
    pip install --no-cache-dir -e .
else
    echo "Error: pyproject.toml not found. Cannot install FedYOLO."
    exit 1
fi

sed -i "s|^BASE = .*|BASE = \"$PATH_CONTAINING_PROJECT\"|" "$CLIENT_CONFIG_FILE"
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

# Clear previous test results directory
echo "Clearing previous test results directory..."
rm -rf "$HOME/test_results"

# First, print the environment summary and client config once at the beginning
echo "Running initial environment summary..."
python3 "$PYTHON_SCRIPT" --dataset_name "${DATASET_NAME_LIST[0]}" --strategy_name "${STRATEGY_LIST[0]}" --print_env_only true

# Loop over datasets and strategies for actual tests
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
                python3 "$PYTHON_SCRIPT" --dataset_name "$DATASET_NAME" --strategy_name "$STRATEGY" --scoring_style "$SCORING_STYLE" --print_summary false --print_env false
                echo ""
            done
        else
            echo "Skipping server-based tests for STRATEGY=${STRATEGY} (contains head/neck/backbone)"
            echo ""
        fi

        # Run client-dependent tests
        for ((CLIENT_NUM=0; CLIENT_NUM<NUM_CLIENTS; CLIENT_NUM++)); do
            for SCORING_STYLE in "${CLIENT_DEPENDENT_STYLES[@]}"; do                
                if should_skip_server "$STRATEGY" && [[ "$SCORING_STYLE" == "server-client" ]]; then
                    echo "Skipping server-client test for STRATEGY=${STRATEGY} (contains head/neck/backbone)"
                    echo ""
                    continue
                fi

                echo "Running client-dependent test: client_num=${CLIENT_NUM}, scoring_style=${SCORING_STYLE}"
                python3 "$PYTHON_SCRIPT" --dataset_name "$DATASET_NAME" --strategy_name "$STRATEGY" --client_num "$CLIENT_NUM" --scoring_style "$SCORING_STYLE" --print_summary false --print_env false
                echo ""
            done
        done
    done
done

# Finally, print the test matrix summary once at the end
echo "Generating final test matrix summary..."
python3 "$PYTHON_SCRIPT" --dataset_name "${DATASET_NAME_LIST[0]}" --strategy_name "${STRATEGY_LIST[0]}" --print_matrix_only true