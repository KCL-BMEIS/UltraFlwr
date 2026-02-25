import torch
from collections import OrderedDict
from typing import Optional, Union, Tuple

import flwr as fl
from flwr.common import parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from ultralytics import YOLO

from fedyolo.train.server_utils import save_model_checkpoint
from fedyolo.config import SPLITS_CONFIG, HOME, get_output_dirs
from fedyolo.train.output_manager import ensure_weights_available


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
        k: v
        for k, v in state_dict.items()
        if not k.startswith(tuple(f"model.{i}" for i in range(9, 24)))
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
        k: v
        for k, v in state_dict.items()
        if k.startswith(tuple(f"model.{i}" for i in range(9, 23)))
    }

    # Head parameters (detection head)
    head_weights = {k: v for k, v in state_dict.items() if k.startswith("model.23")}

    return backbone_weights, neck_weights, head_weights


class BaseYOLOSaveStrategy:
    """Base class for custom FL strategies to save YOLO model checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = (
            f"{HOME}/fedyolo/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        )
        # update_backbone, update_neck, update_head are set as class attributes
        # by create_partial_strategy - do NOT override them here

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters

    def load_and_update_model(self, aggregated_parameters: Parameters) -> YOLO:
        """Load YOLO model and update weights with aggregated parameters."""
        # Ensure base weights are available (ultralytics.settings.weights_dir handles download location)
        output_dirs = get_output_dirs()
        ensure_weights_available("yolo11n.pt", output_dirs["weights_base"])
        net = YOLO(self.model_path)
        current_state_dict = net.model.state_dict()  # type: ignore
        backbone_weights, neck_weights, head_weights = get_section_parameters(
            current_state_dict
        )
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        relevant_keys = []
        for k in sorted(
            current_state_dict.keys()
        ):  # Use sorted() for consistency with client
            if (
                (self.update_backbone and k in backbone_weights)
                or (self.update_neck and k in neck_weights)
                or (self.update_head and k in head_weights)
            ):
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
        net.model.load_state_dict(final_state_dict, strict=True)  # type: ignore
        return net

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(  # type: ignore
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            net = self.load_and_update_model(aggregated_parameters)
            save_model_checkpoint(server_round, model=net.model)

            # Always send back the aggregated parameters (not the full model state dict)
            # This ensures we only send the parameters that were actually aggregated
            return aggregated_parameters, aggregated_metrics

        return aggregated_parameters, aggregated_metrics


def create_partial_strategy(
    base_strategy, name, update_backbone, update_neck, update_head, docstring
):
    """
    Factory function to create partial federated learning strategies.

    Args:
        base_strategy: The base Flower strategy class (e.g., fl.server.strategy.FedAvg)
        name: Name of the strategy class
        update_backbone: Whether to update backbone parameters
        update_neck: Whether to update neck parameters
        update_head: Whether to update head parameters
        docstring: Documentation string for the strategy

    Returns:
        A new strategy class with specified partial update configuration
    """
    return type(
        name,
        (BaseYOLOSaveStrategy, base_strategy),
        {
            "update_backbone": update_backbone,
            "update_neck": update_neck,
            "update_head": update_head,
            "__doc__": docstring,
        },
    )


# Strategy configurations: (name, update_backbone, update_neck, update_head, description)
_STRATEGY_CONFIGS = [
    ("", True, True, True, "all model parameters"),
    ("Head", False, False, True, "detection head only"),
    ("Neck", False, True, False, "neck (SPPF and FPN) only"),
    ("Backbone", True, False, False, "backbone only"),
    ("NeckHead", False, True, True, "neck and head"),
    ("BackboneHead", True, False, True, "backbone and head"),
    ("BackboneNeck", True, True, False, "backbone and neck"),
]

# FedAvg variations
for suffix, update_backbone, update_neck, update_head, description in _STRATEGY_CONFIGS:
    class_name = f"Fed{suffix}Avg"
    docstring = f"Federated averaging of {description}."
    globals()[class_name] = create_partial_strategy(
        fl.server.strategy.FedAvg,
        class_name,
        update_backbone,
        update_neck,
        update_head,
        docstring,
    )

# FedMedian variations
for suffix, update_backbone, update_neck, update_head, description in _STRATEGY_CONFIGS:
    class_name = f"Fed{suffix}Median"
    docstring = f"Federated median of {description}."
    globals()[class_name] = create_partial_strategy(
        fl.server.strategy.FedMedian,
        class_name,
        update_backbone,
        update_neck,
        update_head,
        docstring,
    )

# Export all strategy classes for type checking
# Note: These are dynamically created in the loops above, so we don't list them in __all__
# This prevents false "not present in module" warnings
