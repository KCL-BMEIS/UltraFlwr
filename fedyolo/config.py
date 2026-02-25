# config.py
import os
import yaml
from ultralytics import settings as ultralytics_settings

# Base Configuration
HOME = "/home/localssk23/UltraFlwr"

# Configure Ultralytics to download weights to weights/base/ instead of repo root
ultralytics_settings.update({"weights_dir": f"{HOME}/weights/base"})


def get_nc_from_yaml(yaml_path, task=None):
    """Get number of classes from data.yaml file and update nc dynamically for pose tasks."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Only update nc if the task is 'pose'
    if task == "pose":
        num_classes = len(data.get("names", {}))
        data["nc"] = num_classes  # Update nc dynamically

        # Write the updated data back to the file
        with open(yaml_path, "w") as file:
            yaml.safe_dump(data, file)
    else:
        num_classes = data.get("nc", None)

    return num_classes


def get_nc_from_classification_dataset(data_path):
    """Get number of classes from a classification dataset by checking train directory structure."""
    # For classification datasets, check if we have a partition structure
    if os.path.exists(data_path) and "partitions" in os.listdir(data_path):
        # Look for the first client partition that has a train directory
        partitions_path = os.path.join(data_path, "partitions")
        for partition in os.listdir(partitions_path):
            partition_path = os.path.join(partitions_path, partition)
            if os.path.isdir(partition_path):
                train_path = os.path.join(partition_path, "train")
                if os.path.exists(train_path):
                    # Count the number of class directories
                    classes = [
                        d
                        for d in os.listdir(train_path)
                        if os.path.isdir(os.path.join(train_path, d))
                    ]
                    return len(classes)

    return 0


# Output Directory Configuration
# Set EXPERIMENT_NAME to organize outputs by experiment type
# Examples: "baseline_fedavg", "backbone_only", "full_model_training"
# If not set, defaults to the strategy name from SERVER_CONFIG
# For SSL mode, it's automatically set to "ssl_pretraining"
EXPERIMENT_NAME = (
    "ssl_pretraining"
    if os.getenv("FEDYOLO_SSL_MODE", "false").lower() == "true"
    else None
)  # Will be set dynamically based on strategy if None


def get_output_dirs(experiment_name=None):
    """
    Get output directory paths for the current experiment.

    Args:
        experiment_name: Name of the experiment. If None, uses EXPERIMENT_NAME from config
                        or falls back to SERVER_CONFIG["strategy"]

    Returns:
        dict: Dictionary containing all output directory paths
    """
    global HOME, EXPERIMENT_NAME, SERVER_CONFIG

    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME or SERVER_CONFIG.get(
            "strategy", "default_experiment"
        )

    OUTPUT_ROOT = f"{HOME}/experiments/{experiment_name}"

    return {
        "root": OUTPUT_ROOT,
        "config": f"{OUTPUT_ROOT}/config",
        "logs_server": f"{OUTPUT_ROOT}/logs/server",
        "logs_clients": f"{OUTPUT_ROOT}/logs/clients",
        "logs_testing": f"{OUTPUT_ROOT}/logs/testing",
        "checkpoints_server": f"{OUTPUT_ROOT}/checkpoints/server",
        "checkpoints_clients": f"{OUTPUT_ROOT}/checkpoints/clients",
        "checkpoints_ssl_clients": f"{OUTPUT_ROOT}/checkpoints/ssl_clients",
        "results_metrics": f"{OUTPUT_ROOT}/results/metrics",
        "results_viz": f"{OUTPUT_ROOT}/results/visualizations",
        "metadata": f"{OUTPUT_ROOT}/metadata",
        # Keep base weights in original location (pretrained models)
        "weights_base": f"{HOME}/weights/base",
        # SSL pretrained weights
        "weights_ssl": f"{HOME}/weights/ssl",
    }


# --- Multi-client, multi-task configuration ---

# Specify number of clients per dataset as a dictionary
# Each entry: {"dataset_name": num_clients}
DETECTION_CLIENTS = {"baseline": 1}  # Detection task
SEGMENTATION_CLIENTS = {"seg": 1}  # Segmentation task
POSE_CLIENTS = {"pose": 1}  # Pose estimation task
CLASSIFICATION_CLIENTS = {"mnist": 1}  # Classification task


# Build client specifications for each task type
client_specs = []

# Add detection clients
for ds, n_clients in DETECTION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({"dataset_name": ds, "task": "detect", "client_idx": i})

# Add segmentation clients
for ds, n_clients in SEGMENTATION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({"dataset_name": ds, "task": "segment", "client_idx": i})

# Add pose clients
for ds, n_clients in POSE_CLIENTS.items():
    for i in range(n_clients):
        # Pose dataset has test data in client_2, not client_0
        if ds == "pose":
            client_idx = 2 if i == 0 else i  # Use client_2 for first pose client
        else:
            client_idx = i
        client_specs.append(
            {"dataset_name": ds, "task": "pose", "client_idx": client_idx}
        )

# Add classification clients
for ds, n_clients in CLASSIFICATION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({"dataset_name": ds, "task": "classify", "client_idx": i})

NUM_CLIENTS = len(client_specs)

# Build CLIENT_CONFIG
CLIENT_CONFIG = {}
for cid, spec in enumerate(client_specs):
    dataset_name = spec["dataset_name"]
    task = spec["task"]
    client_idx = spec["client_idx"]
    dataset_path = f"{HOME}/datasets/{dataset_name}"

    # For classification tasks, use data.yaml like other tasks
    if task == "classify":
        data_path = f"{dataset_path}/partitions/client_{client_idx}/data.yaml"
        nc = get_nc_from_classification_dataset(dataset_path)
    else:
        data_yaml = f"{dataset_path}/data.yaml"
        data_path = f"{dataset_path}/partitions/client_{client_idx}/data.yaml"
        nc = get_nc_from_yaml(data_yaml, task=task)  # Pass the task to the function

    CLIENT_CONFIG[cid] = {
        "cid": cid,
        "dataset_name": dataset_name,
        "num_classes": nc,
        "data_path": data_path,
        "task": task,
    }

CLIENT_TASKS = [CLIENT_CONFIG[i]["task"] for i in range(NUM_CLIENTS)]
CLIENT_RATIOS = [1 / NUM_CLIENTS] * NUM_CLIENTS

# For backward compatibility, set the first detection dataset name
DATASET_NAME = list(DETECTION_CLIENTS.keys())[0] if DETECTION_CLIENTS else ""
DATASET_PATH = f"{HOME}/datasets/{DATASET_NAME}"
DATA_YAML = f"{DATASET_PATH}/data.yaml"
NC = get_nc_from_yaml(DATA_YAML) if DATASET_NAME else None

SPLITS_CONFIG = {
    "dataset_name": DATASET_NAME,
    "num_classes": NC,
    "dataset": DATASET_PATH,
    "num_clients": NUM_CLIENTS,
    "ratio": CLIENT_RATIOS,
}

SERVER_CONFIG = {
    "server_address": "0.0.0.0:8080",
    "rounds": 2,
    "sample_fraction": 1.0,
    "min_num_clients": NUM_CLIENTS,
    "max_num_clients": NUM_CLIENTS * 2,  # Adjusted based on number of clients
    "strategy": "FedBackboneAvg",  # Only aggregate backbone (early feature extraction layers)
}

YOLO_CONFIG = {"batch_size": 2, "epochs": 2, "seed_offset": 0}

# SSL (Self-Supervised Learning) Configuration
SSL_CONFIG = {
    "method": "byol",  # Default SSL method: byol, simclr, moco, barlow_twins, vicreg
    "ssl_epochs": 2,  # Number of epochs for SSL pretraining per round (minimal for testing)
    "ssl_batch_size": 64,  # Batch size for SSL training (larger is better for contrastive methods)
    "temperature": 0.5,  # Temperature for contrastive loss (SimCLR, MoCo)
    "projection_dim": 128,  # Dimension of projection head output
    "hidden_dim": 2048,  # Hidden dimension in projection head
    "allow_heterogeneous": False,  # Allow different SSL methods per client (experimental)
    "save_path": f"{HOME}/weights/ssl",  # Where to save SSL-pretrained weights
}

# SSL Server Configuration (for fedyolo-train-ssl)
SSL_SERVER_CONFIG = {
    "server_address": "0.0.0.0:8080",
    "rounds": 2,  # Minimal rounds for testing
    "sample_fraction": 1.0,
    "min_num_clients": NUM_CLIENTS,
    "max_num_clients": NUM_CLIENTS * 2,
    "strategy": "FedBackboneAvg",  # Only aggregate backbone for SSL
}

# Per-client SSL configuration overrides (optional)
# Customize SSL settings for specific clients (e.g., different methods or epochs)
# See documentation for examples
CLIENT_SSL_CONFIG = {}
