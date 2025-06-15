import argparse
import warnings
from collections import OrderedDict
import torch
import flwr as fl
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME
from flwr.common import Context
from flwr.client import ClientApp
from FedYOLO.test.extract_final_save_from_client import extract_results_path

warnings.filterwarnings("ignore", category=UserWarning)

# parser = argparse.ArgumentParser()
# parser.add_argument("--cid", type=int, required=True)
# parser.add_argument("--data_path", type=str, default="./client_0_assets/dummy_data_0/data.yaml")

NUM_CLIENTS = SERVER_CONFIG['max_num_clients']

def train(net, data_path, cid, strategy):
    net.train(data=data_path, epochs=YOLO_CONFIG['epochs'], workers=0, seed=cid+6, batch=YOLO_CONFIG['batch_size'], project=strategy)

# Define get_section_parameters as a standalone function
from typing import Tuple
def get_section_parameters(state_dict: OrderedDict) -> Tuple[dict, dict, dict]:
    """Get parameters for each section of the model."""
    # Backbone parameters (early layers through conv layers)
    # backbone corresponds to:
    # (0): Conv
    # (1): Conv
    # (2): C3k2
    # (3): Conv
    # (4): C3k2
    # (5): Conv
    # (6): C3k2
    # (7): Conv
    # (8): C3k2
    backbone_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith(tuple(f'model.{i}' for i in range(9, 24)))
    }

    # Neck parameters
    # The neck consists of the following layers (by index in the Sequential container):
    # (9): SPPF
    # (10): C2PSA
    # (11): Upsample
    # (12): Concat
    # (13): C3k2
    # (14): Upsample
    # (15): Concat
    # (16): C3k2
    # (17): Conv
    # (18): Concat
    # (19): C3k2
    # (20): Conv
    # (21): Concat
    # (22): C3k2
    neck_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith(tuple(f'model.{i}' for i in range(9, 23)))
    }

    # Head parameters (detection head)
    head_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith('model.23')
    }

    return backbone_weights, neck_weights, head_weights

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path, dataset_name, strategy_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = YOLO()
        self.cid = cid
        self.data_path = data_path
        self.dataset_name=dataset_name
        self.strategy_name=strategy_name

    def get_parameters(self):
        """Get relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()
        # Use the imported function
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

        # Define strategy groups (same as in set_parameters) - Corrected lists
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

        # Determine which parts to send based on strategy
        send_backbone = self.strategy_name in backbone_strategies
        send_neck = self.strategy_name in neck_strategies
        send_head = self.strategy_name in head_strategies

        relevant_parameters = []
        for k, v in current_state_dict.items():
            if (send_backbone and k in backbone_weights) or \
               (send_neck and k in neck_weights) or \
               (send_head and k in head_weights):
                relevant_parameters.append(v.cpu().numpy())
        
        return relevant_parameters

    def set_parameters(self, parameters):
        # Zip server parameters with model state_dict keys
        params_dict = zip(self.net.model.state_dict().keys(), parameters)
        
        # Get current client model state and split into sections
        current_state_dict = self.net.model.state_dict()
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)
        
        # Define strategy groups for backbone, neck, and head updates
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
        
        # Determine which parts to update based on strategy
        update_backbone = self.strategy_name in backbone_strategies
        update_neck = self.strategy_name in neck_strategies
        update_head = self.strategy_name in head_strategies
        
        # Prepare updated weights
        updated_weights = {}
        # Note params_dict is from server
        for k, v in params_dict:
            if (update_backbone and k in backbone_weights) or \
            (update_neck and k in neck_weights) or \
            (update_head and k in head_weights):
                updated_weights[k] = torch.tensor(v)

        # Load updated parameters into the model
        updated_state_dict = OrderedDict(updated_weights)
        self.net.model.load_state_dict(updated_state_dict, strict=False)

    def fit(self, parameters, config):
        if config["server_round"] != 1:
            del self.net
            torch.cuda.empty_cache()
            # get the path of the saved model weight
            logs_path = f"{HOME}/logs/client_{self.cid}_log_{self.dataset_name}_{self.strategy_name}.txt"
            weights_path = extract_results_path(logs_path)
            weights = f"{HOME}/{weights_path}/weights/best.pt"
            print(weights)

            self.net = YOLO(weights)

        self.set_parameters(parameters) # this needs to be modified so we only asign parts of the weights
        train(self.net, self.data_path, self.cid, f"logs/Ultralytics_logs/{self.strategy_name}_{self.dataset_name}_{self.cid}")
        return self.get_parameters(), 10, {}
    
def client_fn(context: Context):
    # args = parser.parse_args()
    # assert args.cid < NUM_CLIENTS
    cid = context.node_config["cid"]
    data_path = context.node_config["data_path"]
    assert cid < NUM_CLIENTS

    return FlowerClient(cid, data_path, SPLITS_CONFIG['dataset_name'], SERVER_CONFIG['strategy']).to_client()

app = ClientApp(
    client_fn,
)
