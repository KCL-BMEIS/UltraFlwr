# UltraFlwr Result Structure

UltraFlwr organizes all experiment outputs in a centralized, hierarchical structure under the `experiments/` directory. This design ensures reproducibility, easy organization, and clean separation of different experiment runs.

## Directory Structure

```
experiments/
└── {experiment_name}/          # Named by strategy (e.g., FedBackboneAvg)
    ├── config/
    │   ├── experiment_config.yaml
    │   └── experiment_config.json
    ├── logs/
    │   ├── server/
    │   │   └── server_{dataset}_{strategy}.log
    │   ├── clients/
    │   │   └── client_{id}/
    │   │       └── training_{dataset}_{strategy}.log
    │   └── testing/
    │       ├── test_run_{timestamp}.log
    │       └── client_{id}_{test_type}_{timestamp}.log
    ├── checkpoints/
    │   ├── server/
    │   │   └── round_{round}_{dataset}_{strategy}.pt
    │   └── clients/
    │       └── client_{id}/
    │           └── train{n}/
    │               ├── weights/
    │               │   ├── best.pt
    │               │   └── last.pt
    │               ├── results.csv
    │               ├── results.png
    │               ├── confusion_matrix.png
    │               └── *_curve.png
    ├── results/
    │   ├── metrics/
    │   │   ├── client_{id}_own_data_{dataset}_{strategy}.csv
    │   │   ├── client_{id}_on_client_{j}_data_{dataset}_{strategy}.csv
    │   │   └── server_on_client_{id}_data_{dataset}_{strategy}.csv
    │   └── visualizations/
    │       ├── client_{id}/
    │       │   └── val_{dataset}/
    │       │       ├── confusion_matrix.png
    │       │       ├── confusion_matrix_normalized.png
    │       │       ├── F1_curve.png
    │       │       ├── P_curve.png
    │       │       ├── PR_curve.png
    │       │       ├── R_curve.png
    │       │       └── val_batch{n}_{labels|pred}.jpg
    │       └── server/
    │           └── val_client_{id}_data/
    └── metadata/
        └── experiment_info.json
```

## Directory Descriptions

### `config/`
Contains snapshots of the experiment configuration for reproducibility:
- **experiment_config.yaml**: Human-readable YAML format
- **experiment_config.json**: Machine-readable JSON format

Both files capture:
- Server configuration (strategy, rounds, etc.)
- Client configurations (datasets, tasks, etc.)
- YOLO training parameters

### `logs/`
Stores all training and testing logs:

#### `server/`
- Server aggregation logs showing federated learning progress
- Records checkpoint saving and round summaries

#### `clients/client_{id}/`
- Individual client training logs
- Contains full YOLO training output including metrics and validation results

#### `testing/`
- Test run summaries
- Individual client evaluation logs with timestamps

### `checkpoints/`
Model weights and training artifacts:

#### `server/`
- Aggregated global model checkpoints after each federated round
- Format: `round_{round}_{dataset}_{strategy}.pt`

#### `clients/client_{id}/train{n}/`
- Client-specific trained models
- Multiple `train` directories due to Ultralytics auto-incrementing (train, train2, train3, etc.)
- Contains:
  - **weights/**: Model checkpoints (best.pt, last.pt)
  - **results.csv**: Training metrics per epoch
  - **results.png**: Training curve visualizations
  - **confusion_matrix.png**: Validation confusion matrices
  - **{metric}_curve.png**: Precision, Recall, F1, mAP curves

### `results/`
Evaluation results from testing:

#### `metrics/`
CSV files containing class-wise and overall metrics:
- Precision, Recall, mAP50, mAP50-95 (for detection/segmentation/pose)
- Top-1, Top-5 accuracy (for classification)

#### `visualizations/`
Visual evaluation outputs:
- Confusion matrices (normalized and unnormalized)
- Precision-Recall curves
- F1, Precision, Recall curves
- Validation batch predictions vs ground truth

### `metadata/`
Experiment metadata:
- **experiment_info.json**: Contains experiment name, start time, and output root path

## Experiment Naming

By default, experiments are named after the federated learning strategy (e.g., `FedBackboneAvg`). You can customize the experiment name by setting `EXPERIMENT_NAME` in `FedYOLO/config.py`:

```python
# FedYOLO/config.py
EXPERIMENT_NAME = "my_custom_experiment"  # Set to None to use strategy name
```
