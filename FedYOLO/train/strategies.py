import time
import torch
from collections import OrderedDict
from typing import Optional, Union, Tuple # Add Tuple

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Parameters, Scalar
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
        for k in sorted(current_state_dict.keys()):  # Use sorted() for consistency with client
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
            
            # For strategies that update all parts (like FedAvg), send back the full model
            # For partial strategies, send back only the relevant parts
            if self.update_backbone and self.update_neck and self.update_head:
                # Send full model parameters
                full_parameters = [val.cpu().numpy() for _, val in net.model.state_dict().items()]
                return ndarrays_to_parameters(full_parameters), aggregated_metrics
            else:
                # Send only relevant partial parameters (same as what was aggregated)
                return aggregated_parameters, aggregated_metrics

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
    