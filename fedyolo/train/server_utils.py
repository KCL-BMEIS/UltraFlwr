import torch

from datetime import datetime

from ultralytics.utils import __version__

from fedyolo.config import SERVER_CONFIG, SPLITS_CONFIG, HOME, get_output_dirs


def save_model_checkpoint(server_round: int, model=None) -> None:
    """Save model training checkpoints with additional metadata."""
    import os

    checkpoint = {
        "epoch": 0,
        "best_fitness": 0,
        "model": model,  # Save the entire model, not just state_dict
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {},
        "train_metrics": {},
        "train_results": {},
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": __version__,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Determine experiment name based on SSL mode
    # If in SSL mode, use "ssl_pretraining" as experiment name
    ssl_mode = os.getenv("FEDYOLO_SSL_MODE", "false").lower() == "true"
    experiment_name = "ssl_pretraining" if ssl_mode else None

    # Use new output directory structure
    output_dirs = get_output_dirs(experiment_name=experiment_name)
    os.makedirs(output_dirs["checkpoints_server"], exist_ok=True)
    ckpt_path = os.path.join(
        output_dirs["checkpoints_server"],
        f"round_{server_round}_{SPLITS_CONFIG['dataset_name']}_{SERVER_CONFIG['strategy']}.pt",
    )
    torch.save(checkpoint, ckpt_path)
    print(f"Saved server checkpoint to: {ckpt_path}")


def write_yolo_config(dataset_name, num_classes=None):
    base_yaml = f"{HOME}/fedyolo/yolo_configs/yolov11.yaml"

    with open(base_yaml, "r") as file:
        base_yaml_content = file.read()

    if num_classes is not None:
        base_yaml_content = base_yaml_content.replace("nc: 80", f"nc: {num_classes}")

    filename = f"{HOME}/fedyolo/yolo_configs/yolo11n_{dataset_name}.yaml"
    with open(filename, "w") as file:
        file.write(base_yaml_content)

    print(f"YAML configuration file '{filename}' has been created.")
