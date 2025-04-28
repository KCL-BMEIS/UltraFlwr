import os

import numpy as np

import flwr as fl
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from ultralytics import YOLO

from FedYOLO.train.server_utils import write_yolo_config
from FedYOLO.train.strategies import (
    FedAvg, FedMedian,
    FedHeadAvg, FedHeadMedian,
    FedNeckAvg, FedNeckMedian,
    FedBackboneAvg, FedBackboneMedian,
    FedNeckHeadAvg, FedNeckHeadMedian,
    FedBackboneHeadAvg, FedBackboneHeadMedian,
    FedBackboneNeckAvg, FedBackboneNeckMedian,
    get_section_parameters
)

from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME


def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"], 
            "server_round": server_round}


def get_parameters(net: YOLO, strategy_name: str) -> list[np.ndarray]:
    """Extract relevant model parameters from YOLO model based on the strategy."""
    current_state_dict = net.model.state_dict()
    backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

    # Define strategy groups (reuse logic similar to client/strategy)
    backbone_strategies = [
        'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
        'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian'
    ]
    neck_strategies = [
        'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
        'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian'
    ]
    head_strategies = [
        'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
        'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian'
    ]

    # Determine which parts are relevant based on strategy
    include_backbone = strategy_name in backbone_strategies
    include_neck = strategy_name in neck_strategies
    include_head = strategy_name in head_strategies

    relevant_parameters = []
    for k, v in current_state_dict.items():
        if (include_backbone and k in backbone_weights) or \
           (include_neck and k in neck_weights) or \
           (include_head and k in head_weights):
            relevant_parameters.append(v.cpu().numpy())
    
    return relevant_parameters


def create_yolo_yaml(dataset_name: str, num_classes: int, task: str) -> YOLO:
    """Initialize YOLO model with the specified dataset, number of classes, and task."""
    write_yolo_config(dataset_name, num_classes)
    yaml_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml"
    if task == "segment":
        return YOLO("yolo11n-seg.pt")
    else:
        return YOLO(yaml_path)

def server_fn(context: Context):
    """Start the FL server with custom strategy."""
    # Make the directory HOME/FedYOLO/yolo_configs if it does not exist
    os.makedirs(f"{HOME}/FedYOLO/yolo_configs", exist_ok=True)

    # Use the first client task as the default for server initialization
    from FedYOLO.config import CLIENT_TASKS
    server_task = CLIENT_TASKS[0] if hasattr(context, 'node_config') and 'task' in context.node_config else CLIENT_TASKS[0]

    # Create dataset specific YOLO yaml
    model = create_yolo_yaml(SPLITS_CONFIG["dataset_name"], SPLITS_CONFIG["num_classes"], server_task)

    # Initialize server side parameters based on the strategy
    strategy_name = SERVER_CONFIG["strategy"]
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
        "FedBackboneNeckMedian": FedBackboneNeckMedian
    }

    # Get the strategy class from config
    if strategy_name not in strategies:
        raise ValueError(
            f"Invalid strategy '{strategy_name}'. Available strategies: {', '.join(strategies.keys())}"
        )
    
    strategy_class = strategies[strategy_name]
    
    # Initialize the strategy
    strategy = strategy_class(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=SERVER_CONFIG["rounds"])

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
