#!/usr/bin/env bash

# This script runs local training and testing on the specified dataset partitions.
# Models are trained and tested on the same dataset partition.
# The script is used to fill the Central Train in Table 2.
# Now supports multiple seeds for reproducible experiments.

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

BASE_PATH="$(pwd)"

echo "Base directory: $BASE_PATH"

DATASET_NAME="polypGen_f"
DATASET_PATHS=("${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_0/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_1/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_2/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/data.yaml")

# Seeds to run experiments with
SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    LOG_DIR="logs_local_train_${DATASET_NAME}_seed_${SEED}"
    mkdir -p "$LOG_DIR"
    
    echo "Starting experiments with seed $SEED..."
    
    for DATASET_PATH in "${DATASET_PATHS[@]}"; do
        # Extract a meaningful name from the dataset path for the log file
        DATASET_BASENAME=$(basename $(dirname "$DATASET_PATH"))
        if [ "$DATASET_BASENAME" = "$DATASET_NAME" ]; then
            DATASET_BASENAME="full_dataset"
        fi
        
        LOG_FILE="$LOG_DIR/train_${DATASET_BASENAME}_seed_${SEED}.log"

        echo "Starting training on $DATASET_PATH with seed $SEED..."
        python3 scripts/central_train_and_test/local_train_and_test.py --data "$DATASET_PATH" --seed "$SEED" | tee "$LOG_FILE"
        echo "Finished training on $DATASET_PATH with seed $SEED."
        echo "---------------------------------------"
    done
    
    echo "Completed all trainings for seed $SEED."
    echo "======================================="
done

echo "All trainings completed for all seeds."