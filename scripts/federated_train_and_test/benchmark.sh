#!/bin/bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# Install FedYOLO from setup.py, uncomment if already installed
# if [[ -f "setup.py" ]]; then
#     echo "Installing FedYOLO package..."
#     pip install --no-cache-dir -e .
# else
#     echo "Error: setup.py not found. Cannot install FedYOLO."
#     exit 1
# fi

# List of datasets and strategies
DATASET_NAME_LIST=("baseline")
# STRATEGY_LIST=(
#     "FedAvg"
#     "FedHeadAvg"
#     "FedNeckAvg"
#     "FedBackboneAvg"
#     "FedNeckHeadAvg"
#     "FedBackboneHeadAvg"
#     "FedBackboneNeckAvg"
#     "FedMedian"
#     "FedHeadMedian"
#     "FedNeckMedian"
#     "FedBackboneMedian"
#     "FedNeckHeadMedian"
#     "FedBackboneHeadMedian"
#     "FedBackboneNeckMedian"
# )
STRATEGY_LIST=("FedBackboneAvg" "FedNeckMedian")

# Partition the data, comment out if already partitioned
# python3 FedYOLO/data_partitioner/fed_split.py >> logs/data_partition_log.txt 2>&1

# Loop over each dataset and strategy
for DATASET_NAME in "${DATASET_NAME_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        
        echo "===================================================================="
        echo "Running with DATASET_NAME=${DATASET_NAME} and STRATEGY=${STRATEGY}"
        echo "===================================================================="
        
        # Modify the config.py file
        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'strategy': .*/    'strategy': '${STRATEGY}',/" $CLIENT_CONFIG_FILE
        
        # Run the base bash file
        bash "scripts/federated_train_and_test/run.sh"

        # newline
        echo ""
        echo ""
        
    done
done
