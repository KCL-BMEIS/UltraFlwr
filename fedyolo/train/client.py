import warnings
import logging
import os
import torch
import flwr as fl
from ultralytics import YOLO
from fedyolo.config import (
    SERVER_CONFIG,
    YOLO_CONFIG,
    HOME,
    NUM_CLIENTS,
    get_output_dirs,
)
from fedyolo.train.output_manager import ensure_weights_available
from flwr.common import Context
from flwr.client import ClientApp
from fedyolo.train.strategies import get_section_parameters

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global cache to track SSL-initialized models (one-time initialization per client)
_client_models = {}


def _load_ssl_model_once(cid, dataset_name, task):
    """
    Load SSL-pretrained model once per client (cached globally).
    This ensures SSL weights are only loaded from disk on the first call.

    Returns:
        YOLO model with SSL-pretrained backbone, or None if no SSL weights available
    """
    # Check if model already initialized for this client
    if cid in _client_models:
        logger.info(f"Client {cid}: Using cached SSL model (already initialized)")
        return _client_models[cid]

    logger.info(f"Client {cid}: First initialization - loading SSL weights from disk")

    # Initialize model config for this client
    output_dirs = get_output_dirs()
    base_weights_dir = output_dirs["weights_base"]

    # Check for client-specific SSL weights first, then fall back to server SSL weights
    client_ssl_weight_file = os.path.join(
        get_output_dirs("ssl_pretraining")["checkpoints_ssl_clients"],
        f"client_{cid}",
        "ssl_backbone.pt",
    )
    server_ssl_weight_file = os.path.join(output_dirs["weights_ssl"], "yolo11n-ssl.pt")

    if os.path.exists(client_ssl_weight_file):
        ssl_weight_file = client_ssl_weight_file
        use_ssl_weights = True
        logger.info(
            f"Client {cid}: Found client-specific SSL weights at {client_ssl_weight_file}"
        )
    elif os.path.exists(server_ssl_weight_file):
        ssl_weight_file = server_ssl_weight_file
        use_ssl_weights = True
        logger.info(
            f"Client {cid}: Using server-aggregated SSL weights from {server_ssl_weight_file}"
        )
    else:
        use_ssl_weights = False
        logger.info(f"Client {cid}: No SSL weights found, will use base weights")

    if not use_ssl_weights:
        # No SSL weights available - return None, caller will load base weights
        _client_models[cid] = None
        return None

    # Load SSL weights
    logger.info(f"Client {cid}: Loading SSL-pretrained weights from {ssl_weight_file}")
    try:
        # Step 1: Load task-specific base model to get correct architecture
        weight_variants = {
            "segment": "yolo11n-seg.pt",
            "pose": "yolo11n-pose.pt",
            "classify": "yolo11n-cls.pt",
        }
        weight_file = weight_variants.get(task, "yolo11n.pt")
        task_weight_path = os.path.join(base_weights_dir, weight_file)

        if os.path.exists(task_weight_path):
            model_path = task_weight_path
            logger.info(
                f"Client {cid}: Using local task-specific weights: {task_weight_path}"
            )
        else:
            # Download to weights/base/ instead of repo root
            model_path = ensure_weights_available(weight_file, base_weights_dir)
            logger.info(
                f"Client {cid}: Downloaded task-specific weights to: {model_path}"
            )

        # Step 2: Load task-specific model (gets correct head architecture)
        net = YOLO(model_path)
        logger.info(f"Client {cid}: Loaded task-specific architecture for {task}")

        # Step 3: Load SSL backbone weights and update only the backbone
        ssl_checkpoint = torch.load(
            ssl_weight_file, map_location="cpu", weights_only=False
        )

        # Extract SSL state dict - handle both client-specific and server checkpoint formats
        if isinstance(ssl_checkpoint, dict) and "backbone_state_dict" in ssl_checkpoint:
            ssl_state_dict = ssl_checkpoint["backbone_state_dict"]
            logger.info(
                f"Client {cid}: Loaded client-specific SSL backbone (method: {ssl_checkpoint.get('method', 'unknown')})"
            )
        elif isinstance(ssl_checkpoint, dict) and "model" in ssl_checkpoint:
            ssl_state_dict = (
                ssl_checkpoint["model"].state_dict()
                if hasattr(ssl_checkpoint["model"], "state_dict")
                else ssl_checkpoint["model"]
            )
        else:
            ssl_state_dict = ssl_checkpoint

        # Get current model state dict
        current_state_dict = net.model.state_dict()  # type: ignore

        # Update only backbone weights (preserve task-specific head)
        updated_state_dict = current_state_dict.copy()
        updated_count = 0
        for key in ssl_state_dict:
            if key.startswith("model.") and key in current_state_dict:
                layer_num_match = key.split(".")[1]
                if layer_num_match.isdigit() and int(layer_num_match) < 10:
                    if current_state_dict[key].shape == ssl_state_dict[key].shape:
                        updated_state_dict[key] = ssl_state_dict[key]
                        updated_count += 1

        # Load updated weights
        net.model.load_state_dict(updated_state_dict, strict=False)  # type: ignore

        logger.info(
            f"Client {cid}: Successfully loaded SSL-pretrained backbone "
            f"({updated_count} layers) with {task}-specific head"
        )

        # Cache the initialized model
        _client_models[cid] = net
        return net

    except Exception as e:
        raise RuntimeError(
            f"Client {cid}: Failed to load SSL weights from {ssl_weight_file}: {e}\n"
            f"Ensure SSL pretraining completed successfully and weights were saved."
        )


def train(net, data_path, cid, strategy, task="detect"):
    """
    Train the YOLO model.

    Args:
        net: YOLO model instance
        data_path: Path to data (YAML file for detect/segment/pose, directory for classify)
        cid: Client ID
        strategy: Strategy directory path
        task: Task type (detect, segment, pose, classify)
    """
    # For classification, YOLO expects a directory path, not a YAML file
    # Convert data.yaml path to directory path if needed
    from pathlib import Path

    if task == "classify" and (
        data_path.endswith(".yaml") or data_path.endswith(".yml")
    ):
        # Extract parent directory from data.yaml path
        data_path = str(Path(data_path).parent)
        logger.info(f"Classification task: using directory path: {data_path}")

    net.train(
        data=data_path,
        epochs=YOLO_CONFIG["epochs"],
        workers=0,
        seed=cid + YOLO_CONFIG["seed_offset"],
        batch=YOLO_CONFIG["batch_size"],
        project=strategy,
    )


class FlowerClient(fl.client.NumPyClient):
    # Define strategy groups as class constants to avoid duplication
    _BACKBONE_STRATEGIES = [
        "FedAvg",
        "FedBackboneAvg",
        "FedBackboneHeadAvg",
        "FedBackboneNeckAvg",
        "FedMedian",
        "FedBackboneMedian",
        "FedBackboneHeadMedian",
        "FedBackboneNeckMedian",
    ]
    _NECK_STRATEGIES = [
        "FedAvg",
        "FedNeckAvg",
        "FedNeckHeadAvg",
        "FedBackboneNeckAvg",
        "FedMedian",
        "FedNeckMedian",
        "FedNeckHeadMedian",
        "FedBackboneNeckMedian",
    ]
    _HEAD_STRATEGIES = [
        "FedAvg",
        "FedHeadAvg",
        "FedNeckHeadAvg",
        "FedBackboneHeadAvg",
        "FedMedian",
        "FedHeadMedian",
        "FedNeckHeadMedian",
        "FedBackboneHeadMedian",
    ]

    def __init__(
        self, cid, data_path, dataset_name, strategy_name, task, net=None, load_ssl=True
    ):
        # If a pre-initialized model is provided, use it (SSL weights already loaded)
        if net is not None:
            self.net = net
            logger.info(
                f"Client {cid}: Using pre-initialized model (SSL weights already loaded)"
            )
        else:
            # Initialize model config for this client
            yaml_path = f"{HOME}/fedyolo/yolo_configs/yolo11n_{dataset_name}.yaml"
            # Define base weights directory using new output structure
            output_dirs = get_output_dirs()
            base_weights_dir = output_dirs["weights_base"]

            # Check for client-specific SSL weights first, then fall back to server SSL weights (only if load_ssl=True)
            use_ssl_weights = False
            if load_ssl:
                client_ssl_weight_file = os.path.join(
                    get_output_dirs("ssl_pretraining")["checkpoints_ssl_clients"],
                    f"client_{cid}",
                    "ssl_backbone.pt",
                )
                server_ssl_weight_file = os.path.join(
                    output_dirs["weights_ssl"], "yolo11n-ssl.pt"
                )

                if os.path.exists(client_ssl_weight_file):
                    ssl_weight_file = client_ssl_weight_file
                    use_ssl_weights = True
                    logger.info(
                        f"Client {cid}: Found client-specific SSL weights at {client_ssl_weight_file}"
                    )
                elif os.path.exists(server_ssl_weight_file):
                    ssl_weight_file = server_ssl_weight_file
                    use_ssl_weights = True
                    logger.info(
                        f"Client {cid}: Using server-aggregated SSL weights from {server_ssl_weight_file}"
                    )
                else:
                    # No SSL weights found - load task-specific weights instead
                    logger.info(
                        f"Client {cid}: No SSL weights found, loading task-specific weights"
                    )
                    weight_variants = {
                        "segment": "yolo11n-seg.pt",
                        "pose": "yolo11n-pose.pt",
                        "classify": "yolo11n-cls.pt",
                    }
                    weight_file = weight_variants.get(task, "yolo11n.pt")

                    if task in ["segment", "pose", "classify"]:
                        model_path = ensure_weights_available(
                            weight_file, base_weights_dir
                        )
                        self.net = YOLO(model_path)
                    else:
                        self.net = YOLO(yaml_path)
            else:
                logger.info(
                    f"Client {cid}: Skipping SSL loading (load_ssl=False) - will receive weights from server"
                )
                # Don't load any weights - just initialize model architecture
                # Server parameters will be applied via set_parameters() in fit()
                weight_variants = {
                    "segment": "yolo11n-seg.pt",
                    "pose": "yolo11n-pose.pt",
                    "classify": "yolo11n-cls.pt",
                }
                weight_file = weight_variants.get(task, "yolo11n.pt")

                # Load model architecture from correct location
                # (ultralytics.settings.weights_dir handles download location)
                if task in ["segment", "pose", "classify"]:
                    model_path = ensure_weights_available(weight_file, base_weights_dir)
                    self.net = YOLO(model_path)
                else:
                    self.net = YOLO(yaml_path)

                logger.info(
                    f"Client {cid}: Initialized {task} model architecture - ready to receive server parameters"
                )

            if use_ssl_weights:
                logger.info(
                    f"Client {cid}: Loading SSL-pretrained weights from {ssl_weight_file}"
                )
                try:
                    # Load SSL-pretrained weights with task-specific head
                    # Step 1: Load task-specific base model to get correct architecture
                    weight_variants = {
                        "segment": "yolo11n-seg.pt",
                        "pose": "yolo11n-pose.pt",
                        "classify": "yolo11n-cls.pt",
                    }
                    weight_file = weight_variants.get(task, "yolo11n.pt")
                    task_weight_path = os.path.join(base_weights_dir, weight_file)

                    # If task-specific weights don't exist locally, download to weights/base/
                    if os.path.exists(task_weight_path):
                        model_path = task_weight_path
                        logger.info(
                            f"Client {cid}: Using local task-specific weights: {task_weight_path}"
                        )
                    else:
                        # Download to weights/base/ instead of repo root
                        model_path = ensure_weights_available(
                            weight_file, base_weights_dir
                        )
                        logger.info(
                            f"Client {cid}: Downloaded task-specific weights to: {model_path}"
                        )

                    # Step 2: Load task-specific model (gets correct head architecture)
                    self.net = YOLO(model_path)
                    logger.info(
                        f"Client {cid}: Loaded task-specific architecture for {task}"
                    )

                    # Step 3: Load SSL backbone weights and update only the backbone
                    ssl_checkpoint = torch.load(
                        ssl_weight_file, map_location="cpu", weights_only=False
                    )

                    # Extract SSL state dict - handle both client-specific and server checkpoint formats
                    if (
                        isinstance(ssl_checkpoint, dict)
                        and "backbone_state_dict" in ssl_checkpoint
                    ):
                        # Client-specific SSL checkpoint format
                        ssl_state_dict = ssl_checkpoint["backbone_state_dict"]
                        logger.info(
                            f"Client {cid}: Loaded client-specific SSL backbone (method: {ssl_checkpoint.get('method', 'unknown')})"
                        )
                    elif isinstance(ssl_checkpoint, dict) and "model" in ssl_checkpoint:
                        # Server aggregated checkpoint format
                        ssl_state_dict = (
                            ssl_checkpoint["model"].state_dict()
                            if hasattr(ssl_checkpoint["model"], "state_dict")
                            else ssl_checkpoint["model"]
                        )
                    else:
                        ssl_state_dict = ssl_checkpoint

                    # Get current model state dict
                    current_state_dict = self.net.model.state_dict()  # type: ignore

                    # Update only backbone weights (preserve task-specific head)
                    updated_state_dict = current_state_dict.copy()
                    updated_count = 0
                    for key in ssl_state_dict:
                        # Only update backbone layers (model.0 through model.9 typically)
                        # Don't update neck (model.10+) or head (model.22/23)
                        if key.startswith("model.") and key in current_state_dict:
                            layer_num_match = key.split(".")[1]
                            if layer_num_match.isdigit() and int(layer_num_match) < 10:
                                # Check shape compatibility
                                if (
                                    current_state_dict[key].shape
                                    == ssl_state_dict[key].shape
                                ):
                                    updated_state_dict[key] = ssl_state_dict[key]
                                    updated_count += 1

                    # Load updated weights
                    self.net.model.load_state_dict(updated_state_dict, strict=False)  # type: ignore

                    logger.info(
                        f"Client {cid}: Successfully loaded SSL-pretrained backbone "
                        f"({updated_count} layers) with {task}-specific head"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Client {cid}: Failed to load SSL weights from {ssl_weight_file}: {e}\n"
                        f"Ensure SSL pretraining completed successfully and weights were saved."
                    )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cid = cid
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.task = task

    def get_parameters(self, config):
        """Get relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()  # type: ignore
        # Use the imported function
        backbone_weights, neck_weights, head_weights = get_section_parameters(
            current_state_dict
        )

        # Define strategy groups (same as in set_parameters) - Corrected lists
        backbone_strategies = self._BACKBONE_STRATEGIES
        neck_strategies = self._NECK_STRATEGIES
        head_strategies = self._HEAD_STRATEGIES

        # Determine which parts to send based on strategy
        send_backbone = self.strategy_name in backbone_strategies
        send_neck = self.strategy_name in neck_strategies
        send_head = self.strategy_name in head_strategies

        # Get all parameters in consistent order (same as set_parameters)
        all_keys = sorted(current_state_dict.keys())
        relevant_parameters = []

        for k in all_keys:
            if (
                (send_backbone and k in backbone_weights)
                or (send_neck and k in neck_weights)
                or (send_head and k in head_weights)
            ):
                relevant_parameters.append(current_state_dict[k].cpu().numpy())

        logger.info(
            f"Client {self.cid} ({self.task}) sending {len(relevant_parameters)} parameters to server"
        )
        return relevant_parameters

    def set_parameters(self, parameters):
        """Set relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()  # type: ignore
        backbone_weights, neck_weights, head_weights = get_section_parameters(
            current_state_dict
        )

        # Use pre-defined strategy groups
        backbone_strategies = self._BACKBONE_STRATEGIES
        neck_strategies = self._NECK_STRATEGIES
        head_strategies = self._HEAD_STRATEGIES

        # Determine which parts to update based on strategy
        update_backbone = self.strategy_name in backbone_strategies
        update_neck = self.strategy_name in neck_strategies
        update_head = self.strategy_name in head_strategies

        # Get relevant keys in consistent order (same as server and get_parameters)
        relevant_keys = []
        for k in sorted(current_state_dict.keys()):
            if (
                (update_backbone and k in backbone_weights)
                or (update_neck and k in neck_weights)
                or (update_head and k in head_weights)
            ):
                relevant_keys.append(k)

        logger.info(f"Strategy: {self.strategy_name}")
        logger.info(f"Parameters received: {len(parameters)}")
        logger.info(f"Expected relevant parameters: {len(relevant_keys)}")
        logger.info(f"Task: {self.task}")

        # Handle architecture mismatches for heterogeneous federated learning
        if len(parameters) != len(relevant_keys):
            logger.warning(
                "Parameter count mismatch. Client architecture may differ from server."
            )
            logger.info(
                "This is expected for heterogeneous federated learning with different task types."
            )
            logger.info(
                "Attempting intelligent parameter matching for compatible layers..."
            )

            # Advanced parameter matching by shape for cross-architecture compatibility
            updated_weights = {}
            server_params_used = [False] * len(parameters)
            matched_count = 0

            # First pass: try to match parameters by shape
            for k in relevant_keys:
                client_shape = current_state_dict[k].shape
                matched = False

                # Look for a server parameter with matching shape
                for i, server_param in enumerate(parameters):
                    if not server_params_used[i] and server_param.shape == client_shape:
                        updated_weights[k] = torch.tensor(server_param)
                        server_params_used[i] = True
                        matched = True
                        matched_count += 1
                        logger.debug(f"✓ Matched {k} with shape {client_shape}")
                        break

                if not matched:
                    # Keep original client parameter for unmatched layers
                    updated_weights[k] = current_state_dict[k]
                    logger.debug(
                        f"✗ No match for {k} with shape {client_shape}, keeping original"
                    )

            logger.info(
                f"Successfully matched {matched_count}/{len(relevant_keys)} parameters"
            )

            if matched_count == 0:
                raise RuntimeError(
                    f"Critical: No parameters could be matched for client {self.cid}. "
                    f"Server has {len(parameters)} parameters, client expects {len(relevant_keys)}. "
                    f"Parameter count mismatch may indicate incompatible architectures. "
                    f"This will prevent federated learning from converging correctly."
                )
            else:
                logger.info(
                    f"Federated learning proceeding with {matched_count} shared parameters"
                )
        else:
            # Perfect match - proceed normally with shape validation
            params_dict = zip(relevant_keys, parameters)
            updated_weights = {}
            for k, v in params_dict:
                expected_shape = current_state_dict[k].shape
                param_array = v

                # Ensure parameter has correct shape
                if param_array.shape != expected_shape:
                    # Try to reshape if possible
                    if param_array.size == torch.Size(expected_shape).numel():
                        logger.info(
                            f"Reshaping parameter {k} from {param_array.shape} to {expected_shape}"
                        )
                        param_array = param_array.reshape(expected_shape)
                    else:
                        # Raise exception instead of silent fallback
                        raise ValueError(
                            f"Parameter reshape impossible for {k}: "
                            f"received shape {param_array.shape} with size {param_array.size}, "
                            f"expected shape {expected_shape} with size {torch.Size(expected_shape).numel()}. "
                            f"This indicates a fundamental architecture mismatch that cannot be resolved. "
                            f"Client {self.cid} cannot proceed with incompatible parameters."
                        )

                updated_weights[k] = torch.tensor(param_array)

        # Load the updated parameters into the model, keeping existing weights for other parts
        final_state_dict = current_state_dict.copy()
        final_state_dict.update(updated_weights)

        self.net.model.load_state_dict(  # type: ignore
            final_state_dict, strict=True
        )  # Use strict=True if all expected keys are present

    def fit(self, parameters, config):
        # For round 2+, update model with aggregated server parameters
        # Keep the same model instance like SSL training does
        if config["server_round"] != 1:
            logger.info(
                f"Client {self.cid}: Updating model with aggregated server parameters for round {config['server_round']}"
            )

        self.set_parameters(parameters)  # Update with aggregated parameters

        # Use new output directory structure for training outputs
        output_dirs = get_output_dirs()
        client_checkpoint_dir = os.path.join(
            output_dirs["checkpoints_clients"], f"client_{self.cid}"
        )
        train(self.net, self.data_path, self.cid, client_checkpoint_dir, self.task)

        # Return only the relevant parameters based on the strategy
        return self.get_parameters(config={}), 10, {}


def client_fn(context: Context):
    """
    Create Flower client instance.

    SSL weights are only loaded ONCE before Round 1 begins.
    After Round 1, clients receive aggregated parameters from the server
    and should NOT reload SSL weights.
    """
    from fedyolo.config import CLIENT_CONFIG

    cid = int(context.node_config.get("cid", 0))
    cfg = CLIENT_CONFIG[cid]
    data_path = context.node_config.get("data_path", cfg["data_path"])
    dataset_name = cfg["dataset_name"]
    task = context.node_config.get("task", cfg["task"])
    assert cid < NUM_CLIENTS

    # Check if this is the very first initialization (before Round 1)
    # We use a simple marker file to track if SSL was already loaded for this run
    output_dirs = get_output_dirs()
    ssl_loaded_marker = os.path.join(
        output_dirs["checkpoints_clients"], f"client_{cid}", ".ssl_loaded"
    )

    # Load SSL only if marker doesn't exist (first time for this training run)
    if not os.path.exists(ssl_loaded_marker):
        logger.info(
            f"Client {cid}: First initialization for this training run - loading SSL weights from disk"
        )
        ssl_model = _load_ssl_model_once(cid, dataset_name, task)

        # Create marker file to indicate SSL weights have been loaded
        os.makedirs(os.path.dirname(ssl_loaded_marker), exist_ok=True)
        with open(ssl_loaded_marker, "w") as f:
            f.write("SSL weights loaded\n")

        return FlowerClient(
            cid, data_path, dataset_name, SERVER_CONFIG["strategy"], task, net=ssl_model
        ).to_client()
    else:
        logger.info(
            f"Client {cid}: SSL weights were already loaded for this training run - using base weights only"
        )
        # Don't load SSL weights - only load base weights
        # The server parameters will be applied via set_parameters() in fit()
        return FlowerClient(
            cid,
            data_path,
            dataset_name,
            SERVER_CONFIG["strategy"],
            task,
            net=None,
            load_ssl=False,
        ).to_client()


app = ClientApp(
    client_fn,
)
