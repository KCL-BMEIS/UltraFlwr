import time
import torch
from collections import OrderedDict
from typing import Optional, Union, Tuple # Add Tuple

import flwr as fl
from flwr.common import parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager

from ultralytics import YOLO

from FedYOLO.train.server_utils import save_model_checkpoint
from FedYOLO.config import SPLITS_CONFIG, HOME

# Define get_section_parameters as a standalone function
def get_section_parameters(state_dict: OrderedDict) -> Tuple[dict, dict, dict]:
    """Get parameters for each section of the model."""
    # Backbone parameters (early layers through conv layers)
    backbone_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith(('model.17', 'model.20', 'model.21', 'model.22', 'model.23'))
    }

    # Neck parameters (SPPF and FPN layers)
    neck_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith(('model.17', 'model.20', 'model.21', 'model.22'))
    }

    # Head parameters (detection head)
    head_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith('model.23')
    }

    return backbone_weights, neck_weights, head_weights

class BaseYOLOSaveStrategy:
    """Base class for custom FL strategies to save YOLO model checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters

    def load_and_update_model(self, aggregated_parameters: Parameters) -> YOLO:
        """Load YOLO model and update weights with aggregated parameters."""
        net = YOLO(self.model_path)
        current_state_dict = net.model.state_dict()
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        relevant_keys = []
        for k in current_state_dict.keys():
            if (self.update_backbone and k in backbone_weights) or \
               (self.update_neck and k in neck_weights) or \
               (self.update_head and k in head_weights):
                relevant_keys.append(k)

        if len(aggregated_ndarrays) != len(relevant_keys):
            strategy_name = self.__class__.__name__
            raise ValueError(
                f"Mismatch in aggregated parameter count for strategy {strategy_name}: "
                f"received {len(aggregated_ndarrays)}, expected {len(relevant_keys)}"
            )

        params_dict = zip(relevant_keys, aggregated_ndarrays)
        updated_weights = {k: torch.tensor(v) for k, v in params_dict}
        final_state_dict = current_state_dict.copy()
        final_state_dict.update(updated_weights)
        net.model.load_state_dict(final_state_dict, strict=True)
        return net

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            net = self.load_and_update_model(aggregated_parameters)
            save_model_checkpoint(server_round, model=net.model)

        return aggregated_parameters, aggregated_metrics


# FedAvg variations
class FedAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of all model parameters."""
    update_backbone = True
    update_neck = True
    update_head = True


class FedHeadAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of detection head only."""
    update_backbone = False
    update_neck = False
    update_head = True

class FedNeckAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of neck (SPPF and FPN) only."""
    update_backbone = False
    update_neck = True
    update_head = False

class FedBackboneAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of backbone only."""
    update_backbone = True
    update_neck = False
    update_head = False

class FedNeckHeadAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of neck and head."""
    update_backbone = False
    update_neck = True
    update_head = True

class FedBackboneHeadAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of neck and head."""
    update_backbone = True
    update_neck = False
    update_head = True

class FedBackboneNeckAvg(BaseYOLOSaveStrategy, fl.server.strategy.FedAvg):
    """Federated averaging of neck and head."""
    update_backbone = True
    update_neck = True
    update_head = False


# FedMedian variations
class FedMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of all model parameters."""
    update_backbone = True
    update_neck = True
    update_head = True

class FedHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of detection head only."""
    update_backbone = False
    update_neck = False
    update_head = True

class FedNeckMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of neck (SPPF and FPN) only."""
    update_backbone = False
    update_neck = True
    update_head = False

class FedBackboneMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of backbone only."""
    update_backbone = True
    update_neck = False
    update_head = False

class FedNeckHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of neck and head."""
    update_backbone = False
    update_neck = True
    update_head = True

class FedBackboneHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of backbone and head."""
    update_backbone = True
    update_neck = False
    update_head = True

class FedBackboneNeckMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
    """Federated median of backbone and neck."""
    update_backbone = True
    update_neck = True
    update_head = False


# class BaseYOLOSaveStrategy(fl.server.strategy.FedAvg):
#     """Base class for custom FL strategies to save YOLO model checkpoints."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"

#     def initialize_parameters(
#         self, client_manager: ClientManager
#     ) -> Optional[Parameters]:
#         """Initialize global model parameters."""
#         # time.sleep(30) # wait for clients to initialise
#         initial_parameters = self.initial_parameters
#         self.initial_parameters = None  # Don't keep initial parameters in memory
#         return initial_parameters

#     def load_and_update_model(self, aggregated_parameters: Parameters) -> YOLO:
#         """Load YOLO model and update weights with aggregated parameters based on strategy."""
#         net = YOLO(self.model_path)
#         current_state_dict = net.model.state_dict()
#         # Call the standalone function
#         backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)
#         aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

#         # Identify the keys corresponding to the parameters received (aggregated)
#         relevant_keys = []
#         for k in current_state_dict.keys():
#             if (self.update_backbone and k in backbone_weights) or \
#                (self.update_neck and k in neck_weights) or \
#                (self.update_head and k in head_weights):
#                 relevant_keys.append(k)

#         # Ensure the number of aggregated parameters matches the number of relevant keys
#         if len(aggregated_ndarrays) != len(relevant_keys):
#             strategy_name = self.__class__.__name__ # Get the name of the current strategy class
#             raise ValueError(
#                 f"Mismatch in aggregated parameter count for strategy {strategy_name}: "
#                 f"received {len(aggregated_ndarrays)}, expected {len(relevant_keys)}"
#             )

#         # Zip the relevant keys with the aggregated parameters
#         params_dict = zip(relevant_keys, aggregated_ndarrays)
        
#         # Prepare updated weights dictionary using only the aggregated parameters
#         updated_weights = {k: torch.tensor(v) for k, v in params_dict}

#         # Create a full state dict for loading, merging aggregated weights with existing ones
#         final_state_dict = current_state_dict.copy()
#         final_state_dict.update(updated_weights)

#         # Load the updated state dict into the model
#         net.model.load_state_dict(final_state_dict, strict=True) # Use strict=True as we are loading a full state dict
#         return net

#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: list[tuple[ClientProxy, FitRes]],
#         failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
#     ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
#         """Aggregate model weights using weighted average and store checkpoint."""
#         aggregated_parameters, aggregated_metrics = super().aggregate_fit(
#             server_round, results, failures
#         )

#         if aggregated_parameters is not None:
#             net = self.load_and_update_model(aggregated_parameters)
#             save_model_checkpoint(server_round, model=net.model)

#         return aggregated_parameters, aggregated_metrics

# # FedAvg variations
# class FedAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of all model parameters."""
#     update_backbone = True
#     update_neck = True
#     update_head = True

# class FedHeadAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of detection head only."""
#     update_backbone = False
#     update_neck = False
#     update_head = True

# class FedNeckAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of neck (SPPF and FPN) only."""
#     update_backbone = False
#     update_neck = True
#     update_head = False

# class FedBackboneAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of backbone only."""
#     update_backbone = True
#     update_neck = False
#     update_head = False

# class FedNeckHeadAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of neck and head."""
#     update_backbone = False
#     update_neck = True
#     update_head = True

# class FedBackboneNeckAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of neck and head."""
#     update_backbone = True
#     update_neck = True
#     update_head = False

# class FedBackboneHeadAvg(BaseYOLOSaveStrategy):
#     """Federated averaging of neck and head."""
#     update_backbone = True
#     update_neck = False
#     update_head = True



# # FedMedian variations
# class FedMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of all model parameters."""
#     update_backbone = True
#     update_neck = True
#     update_head = True

# class FedHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of detection head only."""
#     update_backbone = False
#     update_neck = False
#     update_head = True

# class FedNeckMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of neck (SPPF and FPN) only."""
#     update_backbone = False
#     update_neck = True
#     update_head = False

# class FedBackboneMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of backbone only."""
#     update_backbone = True
#     update_neck = False
#     update_head = False

# class FedNeckHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of neck and head."""
#     update_backbone = False
#     update_neck = True
#     update_head = True

# class FedBackboneHeadMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of neck and head."""
#     update_backbone = True
#     update_neck = False
#     update_head = True

# class FedBackboneNeckMedian(BaseYOLOSaveStrategy, fl.server.strategy.FedMedian):
#     """Federated median of neck and head."""
#     update_backbone = True
#     update_neck = True
#     update_head = False
