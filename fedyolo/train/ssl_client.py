"""
SSL Client Module for Federated Self-Supervised Learning

This module provides a Flower client that performs SSL (self-supervised learning)
training instead of supervised training.
"""

import warnings
import logging
import torch
import flwr as fl
from ultralytics import YOLO
from fedyolo.config import (
    SSL_CONFIG,
    SSL_SERVER_CONFIG,
    CLIENT_SSL_CONFIG,
    NUM_CLIENTS,
    get_output_dirs,
)
from flwr.common import Context
from flwr.client import ClientApp
from fedyolo.train.ssl_trainer import LightlySSLTrainer

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_ssl_config():
    """
    Validate SSL configuration and warn about heterogeneous SSL methods.

    Raises:
        ValueError: If heterogeneous SSL is used without explicit flag
    """
    # Get all SSL methods being used
    methods = set()
    default_method = SSL_CONFIG.get("method", "byol")

    # Check if any clients have custom SSL methods
    for cid, client_config in CLIENT_SSL_CONFIG.items():
        method = client_config.get("method", default_method)
        methods.add(method)

    # If only default method, no client-specific overrides matter
    if len(CLIENT_SSL_CONFIG) == 0:
        methods = {default_method}

    # Check for heterogeneous methods
    if len(methods) > 1:
        if not SSL_CONFIG.get("allow_heterogeneous", False):
            raise ValueError(
                f"""
                Heterogeneous SSL methods detected: {methods}

                Different SSL methods across clients is EXPERIMENTAL and may result in:
                - Training instability
                - Lower performance (15-25% accuracy drop)
                - Slower convergence

                To enable this experimental feature, set:
                SSL_CONFIG["allow_heterogeneous"] = True

                RECOMMENDED: Use the same SSL method for all clients.
                """
            )

        # Warn user
        logger.warning("=" * 80)
        logger.warning("⚠️  EXPERIMENTAL: Heterogeneous SSL methods detected!")
        logger.warning(f"Methods being used: {methods}")
        logger.warning("This may result in degraded performance.")
        logger.warning(
            "Recommended: Use the same SSL method for all clients (e.g., 'byol')"
        )
        logger.warning("=" * 80)

        # Ensure using FedBackboneAvg
        if SSL_SERVER_CONFIG.get("strategy") != "FedBackboneAvg":
            raise ValueError(
                "Heterogeneous SSL requires FedBackboneAvg strategy. "
                f"Current strategy: {SSL_SERVER_CONFIG.get('strategy')}"
            )


class SSLFlowerClient(fl.client.NumPyClient):
    """Flower client for federated self-supervised learning."""

    def __init__(self, cid, data_path, dataset_name, strategy_name, task):
        """
        Initialize SSL client.

        Args:
            cid: Client ID
            data_path: Path to client's dataset
            dataset_name: Name of the dataset
            strategy_name: Federated learning strategy
            task: Task type (detect, segment, pose, classify)
        """
        # Initialize model config for this client
        output_dirs = get_output_dirs()
        base_weights_dir = output_dirs["weights_base"]

        # For SSL: Always use the same base architecture (yolo11n.pt) for all tasks
        # This ensures all clients have matching backbone structure for aggregation
        # Task-specific heads don't matter since SSL only trains the backbone
        logger.info(f"SSL Client {cid}: Loading unified backbone for task '{task}'")
        self.net = YOLO(f"{base_weights_dir}/yolo11n.pt")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cid = cid
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.task = task

        # Get SSL configuration for this client
        self.ssl_config = self._get_ssl_config()

        # Initialize SSL trainer
        self.ssl_trainer = LightlySSLTrainer(self.net, self.ssl_config, self.task)

        logger.info(
            f"SSL Client {cid} initialized with method: {self.ssl_config.get('method')}"
        )

    def _get_ssl_config(self):
        """Get SSL configuration for this client (with overrides if specified)."""
        # Start with global config
        config = SSL_CONFIG.copy()

        # Override with client-specific config if provided
        if self.cid in CLIENT_SSL_CONFIG:
            config.update(CLIENT_SSL_CONFIG[self.cid])

        return config

    def get_parameters(self, config):
        """
        Get backbone parameters for federated aggregation.

        For SSL, we only send backbone weights (feature extractor).
        Uses the same parameter extraction as regular client for compatibility.
        """
        from fedyolo.train.strategies import get_section_parameters

        # Get the full YOLO model state dict
        current_state_dict = self.net.model.state_dict()  # type: ignore

        # Extract backbone weights using same logic as regular client
        backbone_weights, neck_weights, head_weights = get_section_parameters(
            current_state_dict
        )

        # Get all keys in consistent order
        all_keys = sorted(current_state_dict.keys())
        relevant_parameters = []

        # For FedBackboneAvg, only send backbone parameters
        for k in all_keys:
            if k in backbone_weights:
                relevant_parameters.append(current_state_dict[k].cpu().numpy())

        logger.info(
            f"SSL Client {self.cid} ({self.task}) sending {len(relevant_parameters)} backbone parameters"
        )
        return relevant_parameters

    def set_parameters(self, parameters):
        """
        Set backbone parameters from federated aggregation.

        Args:
            parameters: List of numpy arrays containing aggregated backbone weights
        """
        from fedyolo.train.strategies import get_section_parameters

        current_state_dict = self.net.model.state_dict()  # type: ignore
        backbone_weights, neck_weights, head_weights = get_section_parameters(
            current_state_dict
        )

        # Get relevant keys (backbone only for FedBackboneAvg)
        relevant_keys = []
        for k in sorted(current_state_dict.keys()):
            if k in backbone_weights:
                relevant_keys.append(k)

        logger.info(
            f"SSL Client {self.cid}: Setting {len(parameters)} backbone parameters"
        )

        if len(parameters) != len(relevant_keys):
            logger.warning(
                f"Parameter count mismatch: received {len(parameters)}, expected {len(relevant_keys)}"
            )

        # Update backbone weights
        params_dict = zip(relevant_keys, parameters)
        updated_weights = {}
        for k, v in params_dict:
            updated_weights[k] = torch.tensor(v)

        # Load updated parameters into the model
        final_state_dict = current_state_dict.copy()
        final_state_dict.update(updated_weights)
        self.net.model.load_state_dict(final_state_dict, strict=True)  # type: ignore

        logger.info(f"SSL Client {self.cid} updated backbone with aggregated weights")

    def fit(self, parameters, config):
        """
        Perform SSL training for one federated round.

        Args:
            parameters: Aggregated parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        server_round = int(config.get("server_round", 1))
        logger.info(f"SSL Client {self.cid} starting round {server_round}")

        # Update backbone with aggregated weights (except first round)
        if server_round > 1:
            self.set_parameters(parameters)

        # Run SSL pretraining
        self.ssl_trainer.pretrain(
            data_path=self.data_path,
            client_id=self.cid,
            epochs=self.ssl_config.get("ssl_epochs", 20),
        )

        # Get updated backbone parameters (SSL trainer modifies backbone in-place)
        updated_params = self.get_parameters(config={})

        # Return parameters, num examples (dummy value), and empty metrics
        return updated_params, 100, {}


def client_fn(context: Context):
    """
    Create SSL client instance.

    Args:
        context: Flower context containing node configuration

    Returns:
        SSLFlowerClient instance
    """
    from fedyolo.config import CLIENT_CONFIG

    cid = int(context.node_config.get("cid", 0))
    cfg = CLIENT_CONFIG[cid]
    data_path = context.node_config.get("data_path", cfg["data_path"])
    dataset_name = cfg["dataset_name"]
    task = context.node_config.get("task", cfg["task"])

    assert cid < NUM_CLIENTS

    return SSLFlowerClient(
        cid, data_path, dataset_name, SSL_SERVER_CONFIG["strategy"], task
    ).to_client()


# Flower ClientApp for SSL training
app = ClientApp(client_fn)
