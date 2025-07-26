#!/usr/bin/env bash

# This script was used to test one model on each data partition of the dataset.
# as well as the entire dataset.
# This script is used to generate the results in Table 3 of the paper.

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

BASE_PATH="$(pwd)"

echo "Base directory: $BASE_PATH"

# Function to extract global model path from training log files
extract_global_model_path() {
    local seed=$1
    local log_file="${BASE_PATH}/logs_local_train_polypGen_f_seed_${seed}/train_full_dataset_seed_${seed}.log"
    
    if [[ -f "$log_file" ]]; then
        # Extract the path after "Validating" keyword
        local model_path=$(grep "Validating.*\.pt\.\.\." "$log_file" | sed 's/.*Validating \(.*\.pt\)\.\.\..*/\1/' | tail -1)
        if [[ -n "$model_path" ]]; then
            echo "$model_path"
            return 0
        fi
    fi
    echo ""
    return 1
}

DATASET_NAME="polypGen_f"

# Extract global model paths for seeds 0, 1, 2
echo "Extracting global model paths from training logs..."
GLOBAL_MODEL_PATH_SEED_0=$(extract_global_model_path 0)
GLOBAL_MODEL_PATH_SEED_1=$(extract_global_model_path 1)
GLOBAL_MODEL_PATH_SEED_2=$(extract_global_model_path 2)

echo "Found model paths:"
echo "  Seed 0: $GLOBAL_MODEL_PATH_SEED_0"
echo "  Seed 1: $GLOBAL_MODEL_PATH_SEED_1"
echo "  Seed 2: $GLOBAL_MODEL_PATH_SEED_2"

# Use seed 0 as default, or you can modify this logic as needed
GLOBAL_MODEL_PATH="$GLOBAL_MODEL_PATH_SEED_0"
if [[ -z "$GLOBAL_MODEL_PATH" ]]; then
    echo "Warning: Could not extract model path for seed 0, using fallback path"
    GLOBAL_MODEL_PATH="runs/detect/train51/weights/best.pt"
fi

# Store all model paths in an array
GLOBAL_MODEL_PATHS=()
SEEDS=(0 1 2)

for seed in "${SEEDS[@]}"; do
    case $seed in
        0) model_path="$GLOBAL_MODEL_PATH_SEED_0" ;;
        1) model_path="$GLOBAL_MODEL_PATH_SEED_1" ;;
        2) model_path="$GLOBAL_MODEL_PATH_SEED_2" ;;
    esac
    
    if [[ -n "$model_path" ]]; then
        GLOBAL_MODEL_PATHS+=("$model_path")
        echo "Will evaluate seed $seed model: $model_path"
    else
        echo "Warning: No model path found for seed $seed, skipping..."
    fi
done

DATASET_PATHS=("${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_0/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_1/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/partitions/client_2/data.yaml"
          "${BASE_PATH}/datasets/${DATASET_NAME}/data.yaml")
LOG_DIR="logs_local_eval_${DATASET_NAME}"

mkdir -p "$LOG_DIR"

# Loop through each model and test it on each dataset
for i in "${!GLOBAL_MODEL_PATHS[@]}"; do
    GLOBAL_MODEL_PATH="${GLOBAL_MODEL_PATHS[$i]}"
    SEED="${SEEDS[$i]}"
    
    echo ""
    echo "========================================"
    echo "EVALUATING MODEL FROM SEED $SEED"
    echo "Model: $GLOBAL_MODEL_PATH"
    echo "========================================"
    
    for DATASET_PATH in "${DATASET_PATHS[@]}"; do
        # Extract dataset identifier for log file naming
        DATASET_ID=$(echo "$DATASET_PATH" | sed 's|.*/||' | sed 's|\.yaml||')
        if [[ "$DATASET_PATH" == *"client_0"* ]]; then
            DATASET_ID="client_0"
        elif [[ "$DATASET_PATH" == *"client_1"* ]]; then
            DATASET_ID="client_1"
        elif [[ "$DATASET_PATH" == *"client_2"* ]]; then
            DATASET_ID="client_2"
        elif [[ "$DATASET_PATH" == *"/data.yaml" ]]; then
            DATASET_ID="full_dataset"
        fi
        
        LOG_FILE="$LOG_DIR/seed_${SEED}_on_${DATASET_ID}.log"

        echo "Testing seed $SEED model on $DATASET_ID..."
        python3 scripts/central_train_and_test/local_test_only.py --data "$DATASET_PATH" --model "$GLOBAL_MODEL_PATH" | tee "$LOG_FILE"
        echo "Finished testing seed $SEED model on $DATASET_ID."
        echo "---------------------------------------"
    done
done

echo "All trainings completed."