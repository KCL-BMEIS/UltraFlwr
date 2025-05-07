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
from FedYOLO.train.server_utils import write_yolo_config
# Import the function from strategies
from FedYOLO.train.strategies import get_section_parameters

warnings.filterwarnings("ignore", category=UserWarning)


NUM_CLIENTS = SERVER_CONFIG['max_num_clients']

def train(net, data_path, cid, strategy):
    net.train(data=data_path, epochs=YOLO_CONFIG['epochs'], workers=0, seed=cid, 
              batch=YOLO_CONFIG['batch_size'], project=strategy)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path, dataset_name, num_classes, strategy_name, task):
        # Initialize model config for this client
        write_yolo_config(dataset_name, num_classes)
        yaml_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml"
        # Load segmentation weights or detection config
        if task == "segment":
            self.net = YOLO("yolo11n-seg.pt")
        elif task == "pose":
            self.net = YOLO("yolo11n-pose.pt")
        elif task == "classify":
            self.net = YOLO("yolo11n-cls.pt")
        else:
            self.net = YOLO(yaml_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cid = cid
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.task = task

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
        """Set relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()
        # Use the imported function
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

        # Define strategy groups - Corrected lists
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

        # Identify the keys corresponding to the parameters received from the server
        relevant_keys = []
        for k in current_state_dict.keys():
             if (update_backbone and k in backbone_weights) or \
                (update_neck and k in neck_weights) or \
                (update_head and k in head_weights):
                 relevant_keys.append(k)

        # Ensure the number of parameters received matches the number of relevant keys
        if len(parameters) != len(relevant_keys):
             raise ValueError(f"Mismatch in parameter count: received {len(parameters)}, expected {len(relevant_keys)} for strategy {self.strategy_name}")

        # Zip the relevant keys with the received parameters
        params_dict = zip(relevant_keys, parameters)
        
        # Prepare updated weights dictionary using only the received parameters
        updated_weights = {k: torch.tensor(v) for k, v in params_dict}

        # Load the updated parameters into the model, keeping existing weights for other parts
        # Create a full state dict for loading, merging updated weights with existing ones
        final_state_dict = current_state_dict.copy()
        final_state_dict.update(updated_weights)

        self.net.model.load_state_dict(final_state_dict, strict=True) # Use strict=True if all expected keys are present

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

        self.set_parameters(parameters) # Now handles partial updates
        train(self.net, self.data_path, self.cid, f"logs/Ultralytics_logs/{self.strategy_name}_{self.dataset_name}_{self.cid}")
        # Return only the relevant parameters based on the strategy
        return self.get_parameters(), 10, {}
    
def client_fn(context: Context):
    from FedYOLO.config import CLIENT_CONFIG
    cid = context.node_config.get("cid", 0)
    cfg = CLIENT_CONFIG[cid]
    data_path = context.node_config.get("data_path", cfg["data_path"])
    dataset_name = cfg["dataset_name"]
    num_classes = cfg["num_classes"]
    task = context.node_config.get("task", cfg["task"])
    assert cid < NUM_CLIENTS
    return FlowerClient(cid, data_path, dataset_name, num_classes, SERVER_CONFIG['strategy'], task).to_client()

app = ClientApp(
    client_fn,
)