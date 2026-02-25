import os
import numpy as np
from typing import Any

from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from ultralytics import YOLO

from fedyolo.train.server_utils import write_yolo_config
from fedyolo.config import (
    SERVER_CONFIG,
    YOLO_CONFIG,
    SPLITS_CONFIG,
    HOME,
    get_output_dirs,
)
from fedyolo.train.output_manager import ensure_weights_available

# Import strategies dynamically to handle type checking issues with dynamically created classes
from fedyolo.train import strategies as strategies_module  # type: ignore

strategies_dict: dict[str, Any] = {}
for name in [
    "FedAvg",
    "FedMedian",
    "FedHeadAvg",
    "FedHeadMedian",
    "FedNeckAvg",
    "FedNeckMedian",
    "FedBackboneAvg",
    "FedBackboneMedian",
    "FedNeckHeadAvg",
    "FedNeckHeadMedian",
    "FedBackboneHeadAvg",
    "FedBackboneHeadMedian",
    "FedBackboneNeckAvg",
    "FedBackboneNeckMedian",
]:
    strategies_dict[name] = getattr(strategies_module, name, None)

# For convenience, unpack them (type: ignore for each)
FedAvg = strategies_dict["FedAvg"]  # type: ignore
FedMedian = strategies_dict["FedMedian"]  # type: ignore
FedHeadAvg = strategies_dict["FedHeadAvg"]  # type: ignore
FedHeadMedian = strategies_dict["FedHeadMedian"]  # type: ignore
FedNeckAvg = strategies_dict["FedNeckAvg"]  # type: ignore
FedNeckMedian = strategies_dict["FedNeckMedian"]  # type: ignore
FedBackboneAvg = strategies_dict["FedBackboneAvg"]  # type: ignore
FedBackboneMedian = strategies_dict["FedBackboneMedian"]  # type: ignore
FedNeckHeadAvg = strategies_dict["FedNeckHeadAvg"]  # type: ignore
FedNeckHeadMedian = strategies_dict["FedNeckHeadMedian"]  # type: ignore
FedBackboneHeadAvg = strategies_dict["FedBackboneHeadAvg"]  # type: ignore
FedBackboneHeadMedian = strategies_dict["FedBackboneHeadMedian"]  # type: ignore
FedBackboneNeckAvg = strategies_dict["FedBackboneNeckAvg"]  # type: ignore
FedBackboneNeckMedian = strategies_dict["FedBackboneNeckMedian"]  # type: ignore


def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"], "server_round": server_round}


def get_parameters(net: YOLO, strategy_name: str) -> list[np.ndarray]:
    """Extract relevant model parameters from YOLO model based on the strategy."""
    from fedyolo.train.strategies import get_section_parameters

    current_state_dict = net.model.state_dict()  # type: ignore
    backbone_weights, neck_weights, head_weights = get_section_parameters(
        current_state_dict
    )

    # Define strategy parameter filters
    strategy_filters = {
        # FedAvg variations - all parameters
        "FedAvg": {"backbone": True, "neck": True, "head": True},
        "FedHeadAvg": {"backbone": False, "neck": False, "head": True},
        "FedNeckAvg": {"backbone": False, "neck": True, "head": False},
        "FedBackboneAvg": {"backbone": True, "neck": False, "head": False},
        "FedNeckHeadAvg": {"backbone": False, "neck": True, "head": True},
        "FedBackboneHeadAvg": {"backbone": True, "neck": False, "head": True},
        "FedBackboneNeckAvg": {"backbone": True, "neck": True, "head": False},
        # FedMedian variations - same filters as FedAvg
        "FedMedian": {"backbone": True, "neck": True, "head": True},
        "FedHeadMedian": {"backbone": False, "neck": False, "head": True},
        "FedNeckMedian": {"backbone": False, "neck": True, "head": False},
        "FedBackboneMedian": {"backbone": True, "neck": False, "head": False},
        "FedNeckHeadMedian": {"backbone": False, "neck": True, "head": True},
        "FedBackboneHeadMedian": {"backbone": True, "neck": False, "head": True},
        "FedBackboneNeckMedian": {"backbone": True, "neck": True, "head": False},
    }

    # Get filter for this strategy
    if strategy_name not in strategy_filters:
        # Default to all parameters if strategy not found
        filters = {"backbone": True, "neck": True, "head": True}
    else:
        filters = strategy_filters[strategy_name]

    # Get relevant parameters in sorted order (consistent with client)
    relevant_parameters = []
    for k in sorted(current_state_dict.keys()):
        if (
            (filters["backbone"] and k in backbone_weights)
            or (filters["neck"] and k in neck_weights)
            or (filters["head"] and k in head_weights)
        ):
            relevant_parameters.append(current_state_dict[k].cpu().numpy())

    return relevant_parameters


def create_yolo_yaml(dataset_name: str, num_classes: int, task: str) -> YOLO:
    """Initialize YOLO model with the specified dataset, number of classes, and task."""
    import torch

    write_yolo_config(dataset_name, num_classes)
    yaml_path = f"{HOME}/fedyolo/yolo_configs/yolo11n_{dataset_name}.yaml"
    # Use new output directory structure
    output_dirs = get_output_dirs()
    base_weights_dir = output_dirs["weights_base"]
    ssl_weights_dir = output_dirs["weights_ssl"]

    # Check if server SSL weights exist
    server_ssl_weight_file = os.path.join(ssl_weights_dir, "yolo11n-ssl.pt")
    use_ssl_weights = os.path.exists(server_ssl_weight_file)

    if task == "segment":
        seg_weights = ensure_weights_available("yolo11n-seg.pt", base_weights_dir)
        base_model = YOLO(seg_weights)
    else:
        base_weights = ensure_weights_available("yolo11n.pt", base_weights_dir)
        base_model = YOLO(base_weights)
        # Override with custom config (ultralytics.settings.weights_dir handles download location)
        base_model.model = YOLO(yaml_path).model  # type: ignore

    # Load server-aggregated SSL weights if available
    if use_ssl_weights:
        print(
            f"Server: Loading server-aggregated SSL weights from {server_ssl_weight_file}"
        )
        ssl_checkpoint = torch.load(
            server_ssl_weight_file, map_location="cpu", weights_only=False
        )

        # Extract SSL state dict (server aggregated format)
        if isinstance(ssl_checkpoint, dict) and "model" in ssl_checkpoint:
            ssl_state_dict = (
                ssl_checkpoint["model"].state_dict()
                if hasattr(ssl_checkpoint["model"], "state_dict")
                else ssl_checkpoint["model"]
            )
        else:
            ssl_state_dict = ssl_checkpoint

        # Get current model state dict
        current_state_dict = base_model.model.state_dict()

        # Update only backbone weights (preserve task-specific head)
        updated_state_dict = current_state_dict.copy()
        updated_count = 0
        for key in ssl_state_dict:
            # Only update backbone layers (model.0 through model.9)
            if key.startswith("model.") and key in current_state_dict:
                layer_num_match = key.split(".")[1]
                if layer_num_match.isdigit() and int(layer_num_match) < 10:
                    if current_state_dict[key].shape == ssl_state_dict[key].shape:
                        updated_state_dict[key] = ssl_state_dict[key]
                        updated_count += 1

        # Load updated weights
        base_model.model.load_state_dict(updated_state_dict, strict=False)
        print(
            f"Server: Successfully loaded server-aggregated SSL backbone ({updated_count} layers)"
        )

    return base_model


def server_fn(_context: Context):
    """Start the FL server with custom strategy."""
    # Make the directory HOME/FedYOLO/yolo_configs if it does not exist
    os.makedirs(f"{HOME}/fedyolo/yolo_configs", exist_ok=True)

    # Use detection model as the base architecture for all clients
    # This ensures consistent parameter structure across different tasks
    base_task = "detect"  # Use detection as base architecture

    # Create dataset specific YOLO yaml using detection task
    dataset_name: str = str(SPLITS_CONFIG["dataset_name"])
    num_classes: int = int(SPLITS_CONFIG["num_classes"])
    model = create_yolo_yaml(dataset_name, num_classes, base_task)

    # Initialize server side parameters based on the strategy
    strategy_name: str = str(SERVER_CONFIG["strategy"])

    # Initialize parameters with server model
    initial_parameters = ndarrays_to_parameters(get_parameters(model, strategy_name))

    # Map of available strategies
    strategies = {
        # FedAvg variations
        "FedAvg": FedAvg,
        "FedHeadAvg": FedHeadAvg,
        "FedNeckAvg": FedNeckAvg,
        "FedBackboneAvg": FedBackboneAvg,
        "FedNeckHeadAvg": FedNeckHeadAvg,
        "FedBackboneHeadAvg": FedBackboneHeadAvg,
        "FedBackboneNeckAvg": FedBackboneNeckAvg,
        # FedMedian variations
        "FedMedian": FedMedian,
        "FedHeadMedian": FedHeadMedian,
        "FedNeckMedian": FedNeckMedian,
        "FedBackboneMedian": FedBackboneMedian,
        "FedNeckHeadMedian": FedNeckHeadMedian,
        "FedBackboneHeadMedian": FedBackboneHeadMedian,
        "FedBackboneNeckMedian": FedBackboneNeckMedian,
    }

    # Get the strategy class from config
    if strategy_name not in strategies:
        raise ValueError(
            f"Invalid strategy '{strategy_name}'. Available strategies: {', '.join(strategies.keys())}"
        )

    strategy_class = strategies[strategy_name]

    # Initialize the strategy
    sample_fraction: float = float(SERVER_CONFIG["sample_fraction"])
    min_num_clients: int = int(SERVER_CONFIG["min_num_clients"])
    num_rounds: int = int(SERVER_CONFIG["rounds"])

    strategy = strategy_class(
        fraction_fit=sample_fraction,
        min_fit_clients=min_num_clients,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
