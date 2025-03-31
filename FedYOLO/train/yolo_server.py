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
    FedBackboneNeckAvg, FedBackboneNeckMedian
)

from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME


def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"], 
            "server_round": server_round}


def get_parameters(net: YOLO) -> list[np.ndarray]:
    """Extract model parameters from YOLO model."""
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]


def create_yolo_yaml(dataset_name: str, num_classes: int) -> YOLO:
    """Initialize YOLO model with the specified dataset and number of classes."""
    write_yolo_config(dataset_name, num_classes)
    return YOLO(f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml")

def server_fn(context: Context):
    """Start the FL server with custom strategy."""
    # Make the directory HOME/FedYOLO/yolo_configs if it does not exist
    os.makedirs(f"{HOME}/FedYOLO/yolo_configs", exist_ok=True)

    # Create dataset specific YOLO yaml
    create_yolo_yaml(SPLITS_CONFIG["dataset_name"], SPLITS_CONFIG["num_classes"])

    # Initialize server side parameters
    initial_parameters = ndarrays_to_parameters(get_parameters(YOLO()))

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
    strategy_name = SERVER_CONFIG["strategy"]
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



# def main() -> None:
#     """Start the FL server with custom strategy."""
#     # Make the directory HOME/FedYOLO/yolo_configs if it does not exist
#     os.makedirs(f"{HOME}/FedYOLO/yolo_configs", exist_ok=True)

#     # Create dataset specific YOLO yaml
#     create_yolo_yaml(SPLITS_CONFIG["dataset_name"], SPLITS_CONFIG["num_classes"])

#     # Initialize server side parameters
#     initial_parameters = ndarrays_to_parameters(get_parameters(YOLO()))

#     # Map of available strategies
#     strategies = {
#         # FedAvg variations
#         "FedAvg": FedAvg,
#         "FedHeadAvg": FedHeadAvg,
#         "FedNeckAvg": FedNeckAvg,
#         "FedBackboneAvg": FedBackboneAvg,
#         "FedNeckHeadAvg": FedNeckHeadAvg,
#         "FedBackboneHeadAvg": FedBackboneHeadAvg,
#         "FedBackboneNeckAvg": FedBackboneNeckAvg,
        
#         # FedMedian variations
#         "FedMedian": FedMedian,
#         "FedHeadMedian": FedHeadMedian,
#         "FedNeckMedian": FedNeckMedian,
#         "FedBackboneMedian": FedBackboneMedian,
#         "FedNeckHeadMedian": FedNeckHeadMedian,
#         "FedBackboneHeadMedian": FedBackboneHeadMedian,
#         "FedBackboneNeckMedian": FedBackboneNeckMedian
#     }

#     # Get the strategy class from config
#     strategy_name = SERVER_CONFIG["strategy"]
#     if strategy_name not in strategies:
#         raise ValueError(
#             f"Invalid strategy '{strategy_name}'. Available strategies: {', '.join(strategies.keys())}"
#         )
    
#     strategy_class = strategies[strategy_name]
    
#     # Initialize the strategy
#     strategy = strategy_class(
#         fraction_fit=SERVER_CONFIG["sample_fraction"],
#         min_fit_clients=SERVER_CONFIG["min_num_clients"],
#         on_fit_config_fn=fit_config,
#         initial_parameters=initial_parameters,
#     )

#     # Start Flower server
#     fl.server.start_server(
#         server_address=SERVER_CONFIG["server_address"],
#         config=fl.server.ServerConfig(num_rounds=SERVER_CONFIG["rounds"]),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()
    
