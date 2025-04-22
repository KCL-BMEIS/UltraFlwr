#!/usr/bin/env bash

# This script runs local training and testing on the specified dataset partitions.
# Models are trained and tested on the same dataset partition.
# The script is used to fill the Central Train in Table 2.

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

# Install FedYOLO from setup.py, uncomment if already installed
# if [[ -f "setup.py" ]]; then
#     echo "Installing FedYOLO package..."
#     pip install --no-cache-dir -e .
# else
#     echo "Error: setup.py not found. Cannot install FedYOLO."
#     exit 1
# fi

BASE_PATH="$(pwd)"

echo "Base directory: $BASE_PATH"

DATASET_NAME="baseline"
DATASET_PATHS=("${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_0/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_1/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_2/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/data.yaml")
LOG_DIR="logs_local_train_${DATASET_NAME}"

mkdir -p "$LOG_DIR"

for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    LOG_FILE="$LOG_DIR/train_$(echo "$DATASET_PATH" | sed 's|/|_|g').log"

    echo "Starting training on $DATASET_PATH..."
    python3 scripts/central_train_and_test/local_train_and_test.py --data "$DATASET_PATH" | tee "$LOG_FILE"
    echo "Finished training on $DATASET_PATH."
    echo "---------------------------------------"
done

echo "All trainings completed."