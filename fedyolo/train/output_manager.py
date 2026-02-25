"""
Output directory management for federated learning experiments.

This module handles creation and initialization of output directories,
configuration snapshots, experiment metadata, and weight file management.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def weights_dir_context(base_weights_dir=None):
    """
    Context manager that temporarily changes to weights directory.

    Use this when loading YOLO models from YAML configs to ensure any
    automatic weight downloads go to weights/base/ instead of repo root.

    Args:
        base_weights_dir: Directory for weights. If None, uses config default.

    Usage:
        with weights_dir_context():
            model = YOLO("yolo11n.yaml")
    """
    if base_weights_dir is None:
        from fedyolo.config import get_output_dirs

        base_weights_dir = get_output_dirs()["weights_base"]

    # Ensure directory exists
    os.makedirs(base_weights_dir, exist_ok=True)

    original_cwd = os.getcwd()
    try:
        os.chdir(base_weights_dir)
        yield base_weights_dir
    finally:
        os.chdir(original_cwd)


def initialize_experiment_output(experiment_name=None, save_config=True):
    """
    Initialize output directory structure for an experiment.

    Creates all necessary subdirectories and optionally saves a snapshot
    of the current configuration for reproducibility.

    Args:
        experiment_name: Name of the experiment. If None, uses config default
        save_config: Whether to save config snapshot

    Returns:
        dict: Dictionary of output directory paths
    """
    from fedyolo.config import (
        get_output_dirs,
        SERVER_CONFIG,
        CLIENT_CONFIG,
        YOLO_CONFIG,
    )

    # Get output directories
    output_dirs = get_output_dirs(experiment_name)

    # Create all directories
    for key, path in output_dirs.items():
        if key != "weights_base":  # Don't create weights/base, it should already exist
            os.makedirs(path, exist_ok=True)

    # Create client-specific subdirectories
    for cid in CLIENT_CONFIG.keys():
        client_log_dir = os.path.join(output_dirs["logs_clients"], f"client_{cid}")
        client_checkpoint_dir = os.path.join(
            output_dirs["checkpoints_clients"], f"client_{cid}"
        )
        client_viz_dir = os.path.join(output_dirs["results_viz"], f"client_{cid}")

        os.makedirs(client_log_dir, exist_ok=True)
        os.makedirs(client_checkpoint_dir, exist_ok=True)
        os.makedirs(client_viz_dir, exist_ok=True)

    # Save configuration snapshot
    if save_config:
        save_experiment_config(output_dirs, SERVER_CONFIG, CLIENT_CONFIG, YOLO_CONFIG)

    # Save experiment metadata
    save_experiment_metadata(output_dirs, experiment_name)

    print(f"Initialized experiment output directories at: {output_dirs['root']}")

    return output_dirs


def save_experiment_config(output_dirs, server_config, client_config, yolo_config):
    """
    Save a snapshot of experiment configuration for reproducibility.

    Args:
        output_dirs: Dictionary of output paths
        server_config: Server configuration dict
        client_config: Client configuration dict
        yolo_config: YOLO training configuration dict
    """
    config_dir = output_dirs["config"]

    # Save as YAML for readability
    config_snapshot = {
        "server": server_config,
        "clients": client_config,
        "yolo": yolo_config,
    }

    config_path = os.path.join(config_dir, "experiment_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

    # Also save as JSON for easier programmatic access
    config_json_path = os.path.join(config_dir, "experiment_config.json")
    with open(config_json_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)


def save_experiment_metadata(output_dirs, experiment_name):
    """
    Save experiment metadata (timestamp, name, etc.).

    Args:
        output_dirs: Dictionary of output paths
        experiment_name: Name of the experiment
    """
    from fedyolo.config import SERVER_CONFIG

    metadata = {
        "experiment_name": experiment_name or SERVER_CONFIG.get("strategy", "default"),
        "start_time": datetime.now().isoformat(),
        "output_root": output_dirs["root"],
    }

    metadata_path = os.path.join(output_dirs["metadata"], "experiment_info.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_client_log_path(output_dirs, client_id, dataset_name, strategy_name):
    """
    Get the log file path for a specific client.

    Args:
        output_dirs: Dictionary of output paths
        client_id: Client ID
        dataset_name: Dataset name for this client
        strategy_name: Strategy name

    Returns:
        str: Path to client log file
    """
    log_dir = os.path.join(output_dirs["logs_clients"], f"client_{client_id}")
    log_filename = f"training_{dataset_name}_{strategy_name}.log"
    return os.path.join(log_dir, log_filename)


def get_server_log_path(output_dirs, dataset_name, strategy_name):
    """
    Get the log file path for the server.

    Args:
        output_dirs: Dictionary of output paths
        dataset_name: Dataset name
        strategy_name: Strategy name

    Returns:
        str: Path to server log file
    """
    log_filename = f"server_{dataset_name}_{strategy_name}.log"
    return os.path.join(output_dirs["logs_server"], log_filename)


def get_test_log_path(output_dirs, test_type, client_id=None, timestamp=None):
    """
    Get the log file path for testing.

    Args:
        output_dirs: Dictionary of output paths
        test_type: Type of test (e.g., 'own_data', 'cross_eval', 'test_run')
        client_id: Client ID (if applicable)
        timestamp: Timestamp string (if applicable)

    Returns:
        str: Path to test log file
    """
    if client_id is not None:
        log_filename = f"client_{client_id}_{test_type}"
    else:
        log_filename = test_type

    if timestamp:
        log_filename += f"_{timestamp}"

    log_filename += ".log"

    return os.path.join(output_dirs["logs_testing"], log_filename)


def get_checkpoint_path(
    output_dirs, round_num, dataset_name, strategy_name, is_server=True, client_id=None
):
    """
    Get the checkpoint file path.

    Args:
        output_dirs: Dictionary of output paths
        round_num: Training round number
        dataset_name: Dataset name
        strategy_name: Strategy name
        is_server: Whether this is a server checkpoint
        client_id: Client ID (required if is_server=False)

    Returns:
        str: Path to checkpoint file
    """
    if is_server:
        checkpoint_filename = f"round_{round_num}_{dataset_name}_{strategy_name}.pt"
        return os.path.join(output_dirs["checkpoints_server"], checkpoint_filename)
    else:
        if client_id is None:
            raise ValueError("client_id must be provided for client checkpoints")
        checkpoint_dir = os.path.join(
            output_dirs["checkpoints_clients"], f"client_{client_id}"
        )
        checkpoint_filename = f"round_{round_num}_{dataset_name}_{strategy_name}.pt"
        return os.path.join(checkpoint_dir, checkpoint_filename)


def get_results_path(output_dirs, result_type, client_id=None, eval_type=None):
    """
    Get the results file path.

    Args:
        output_dirs: Dictionary of output paths
        result_type: Type of result (e.g., 'training_metrics', 'evaluation_metrics')
        client_id: Client ID (if applicable)
        eval_type: Evaluation type (e.g., 'own_data', 'cross_eval')

    Returns:
        str: Path to results file
    """
    if client_id is not None:
        result_filename = f"client_{client_id}_{result_type}"
    else:
        result_filename = result_type

    if eval_type:
        result_filename += f"_{eval_type}"

    result_filename += ".csv"

    return os.path.join(output_dirs["results_metrics"], result_filename)


def ensure_weights_available(weight_file, base_weights_dir=None):
    """
    Ensure YOLO weights are available in the correct directory.

    If weights don't exist in base_weights_dir, downloads them there
    instead of polluting the repository root.

    Args:
        weight_file: Weight file name (e.g., "yolo11n.pt", "yolo11n-seg.pt")
        base_weights_dir: Directory for weights. If None, uses config default.

    Returns:
        str: Full path to the weight file in base_weights_dir
    """
    if base_weights_dir is None:
        from fedyolo.config import get_output_dirs

        base_weights_dir = get_output_dirs()["weights_base"]

    # Ensure directory exists
    os.makedirs(base_weights_dir, exist_ok=True)

    target_path = os.path.join(base_weights_dir, weight_file)

    if os.path.exists(target_path):
        logger.debug(f"Weights already available: {target_path}")
        return target_path

    logger.info(f"Downloading {weight_file} to {base_weights_dir}...")

    # Import here to avoid circular imports
    from ultralytics import YOLO

    # Get original working directory
    original_cwd = os.getcwd()

    try:
        # Change to weights directory before loading
        # This makes YOLO download to the correct location
        os.chdir(base_weights_dir)

        # Load model (triggers download to current directory)
        YOLO(weight_file)

        # Verify the file was downloaded
        if os.path.exists(target_path):
            logger.info(f"Successfully downloaded: {target_path}")
        else:
            # YOLO might have downloaded to a different name, check for the file
            downloaded_file = os.path.join(base_weights_dir, weight_file)
            if not os.path.exists(downloaded_file):
                raise RuntimeError(f"Failed to download {weight_file}")

    finally:
        # Always restore original working directory
        os.chdir(original_cwd)

    return target_path
