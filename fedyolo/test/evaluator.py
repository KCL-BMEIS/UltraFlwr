from ultralytics import YOLO
from fedyolo.test.utils import extract_results_path

from fedyolo.config import HOME, SERVER_CONFIG, get_output_dirs
from fedyolo.train.output_manager import ensure_weights_available

import pandas as pd
import os
import torch

import argparse


def load_fl_trained_model(weights_path, task, dataset_name=None, output_dirs=None):
    """
    Load a federated learning-trained model with the correct task-specific architecture.

    Since FL training (especially backbone-only strategies like FedBackboneAvg) may save
    models as DetectionModel regardless of the original task, this function:
    1. Loads the task-specific base model
    2. Loads the FL-trained backbone weights
    3. Combines them to create a properly task-configured model

    Args:
        weights_path: Path to FL-trained model weights (.pt file)
        task: Task type ('detect', 'segment', 'pose', 'classify')
        dataset_name: Dataset name (optional, for finding task-specific YAML)
        output_dirs: Output directories dict (optional)

    Returns:
        YOLO model with correct task architecture and FL-trained backbone
    """
    if output_dirs is None:
        output_dirs = get_output_dirs()

    base_weights_dir = output_dirs["weights_base"]

    # For pose and custom tasks, try to use task-specific YAML if available
    # This handles cases where the model has custom keypoint configurations
    if dataset_name and task in ["pose"]:
        yaml_path = f"{HOME}/fedyolo/yolo_configs/yolo11n_{dataset_name}.yaml"
        if os.path.exists(yaml_path):
            # Load from YAML (ultralytics.settings.weights_dir handles download location)
            model = YOLO(yaml_path, task=task)
        else:
            # Fallback to base weights
            weight_variants = {
                "segment": "yolo11n-seg.pt",
                "pose": "yolo11n-pose.pt",
                "classify": "yolo11n-cls.pt",
                "detect": "yolo11n.pt",
            }
            base_weight_file = weight_variants.get(task, "yolo11n.pt")

            # Ensure weights are in weights/base/, not repo root
            model_path = ensure_weights_available(base_weight_file, base_weights_dir)
            model = YOLO(model_path, task=task)
    else:
        # Map task to base weight file
        weight_variants = {
            "segment": "yolo11n-seg.pt",
            "pose": "yolo11n-pose.pt",
            "classify": "yolo11n-cls.pt",
            "detect": "yolo11n.pt",
        }
        base_weight_file = weight_variants.get(task, "yolo11n.pt")

        # Ensure weights are in weights/base/, not repo root
        model_path = ensure_weights_available(base_weight_file, base_weights_dir)
        model = YOLO(model_path, task=task)

    # Load FL-trained weights
    fl_checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Extract model state dict from checkpoint
    if isinstance(fl_checkpoint, dict) and "model" in fl_checkpoint:
        fl_state_dict = (
            fl_checkpoint["model"].state_dict()
            if hasattr(fl_checkpoint["model"], "state_dict")
            else fl_checkpoint["model"]
        )
    else:
        fl_state_dict = fl_checkpoint

    # Get current model state dict
    current_state_dict = model.model.state_dict()

    # Update with FL-trained weights (typically backbone only for FedBackboneAvg)
    # Only update keys that exist in both state dicts
    updated_state_dict = current_state_dict.copy()
    for key in fl_state_dict:
        if key in current_state_dict:
            # Check if shapes match before updating
            if current_state_dict[key].shape == fl_state_dict[key].shape:
                updated_state_dict[key] = fl_state_dict[key]

    # Load updated weights
    model.model.load_state_dict(updated_state_dict, strict=False)

    return model


def safe_save_csv(table, filename, description=""):
    """Safely save CSV file with error handling and fallback location"""
    try:
        # Ensure results directory exists
        results_dir = os.path.dirname(filename)
        os.makedirs(results_dir, exist_ok=True)

        # Try to save the CSV
        table.to_csv(filename, index=True, index_label="class")
        print(f"✓ Saved {description}: {filename}")

        # Verify the file was actually created and get its size
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  File verified: {filename} ({file_size} bytes)")
        else:
            print(f"  ⚠ Warning: File not found after save: {filename}")

    except PermissionError:
        # Fallback to /tmp if permission denied
        fallback_filename = filename.replace(f"{HOME}/results", "/tmp/results")
        fallback_dir = os.path.dirname(fallback_filename)
        os.makedirs(fallback_dir, exist_ok=True)
        table.to_csv(fallback_filename, index=True, index_label="class")
        print(f"⚠ Permission denied for {filename}")
        print(f"✓ Saved {description} to fallback location: {fallback_filename}")

    except Exception as e:
        print(f"✗ Error saving {description} to {filename}: {str(e)}")


def list_csv_files(output_dirs=None):
    """List all CSV files in the results directory for debugging"""
    if output_dirs is None:
        output_dirs = get_output_dirs()
    results_dir = output_dirs["results_metrics"]
    print(f"\n=== CSV files in {results_dir} ===")
    try:
        if os.path.exists(results_dir):
            csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
            if csv_files:
                for csv_file in sorted(csv_files):
                    file_path = os.path.join(results_dir, csv_file)
                    file_size = os.path.getsize(file_path)
                    print(f"  {csv_file} ({file_size} bytes)")
            else:
                print("  No CSV files found")
        else:
            print(f"  Directory {results_dir} does not exist")
    except Exception as e:
        print(f"  Error listing files: {str(e)}")
    print("=" * 50)


def _get_task_metrics(results, task):
    """Extract metrics and mean results for a given task type."""
    if task == "detect":
        metrics = results.box
        mean_fn = results.box.mean_results
    elif task == "segment":
        metrics = results.seg
        mean_fn = results.seg.mean_results
    elif task == "pose":
        metrics = results.pose
        mean_fn = results.pose.mean_results
    else:
        raise ValueError(f"Invalid task: {task}")

    precision_values = metrics.p
    recall_values = metrics.r
    ap50_values = metrics.ap50
    ap50_95_values = metrics.ap
    mp, mr, map50, map5095 = mean_fn()

    return (
        precision_values,
        recall_values,
        ap50_values,
        ap50_95_values,
        mp,
        mr,
        map50,
        map5095,
    )


def load_head_only_weights(server_model, base_weights_dir):
    """
    Load detection head weights from server model into a normal YOLO model.

    Args:
        server_model: YOLO model with server weights
        base_weights_dir: Directory containing base YOLO weights

    Returns:
        YOLO model with detection head from server and base weights for other parts
    """
    normal_model = YOLO(f"{base_weights_dir}/yolo11n.pt")
    detection_weights = {  # type: ignore
        k: v
        for k, v in server_model.model.state_dict().items()  # type: ignore
        if k.startswith("model.detect")
    }
    state_dict = normal_model.model.state_dict()  # type: ignore
    normal_model.model.load_state_dict(  # type: ignore
        {**state_dict, **detection_weights}, strict=False
    )
    return normal_model


def get_classwise_results_table(results, task):
    if task in ["detect", "segment", "pose"]:
        (
            precision_values,
            recall_values,
            ap50_values,
            ap50_95_values,
            mp,
            mr,
            map50,
            map5095,
        ) = _get_task_metrics(results, task)

        num_classes = min(len(results.names), len(precision_values))
        class_wise_results = {
            "precision": {
                results.names[idx]: precision_values[idx] for idx in range(num_classes)
            },
            "recall": {
                results.names[idx]: recall_values[idx] for idx in range(num_classes)
            },
            "mAP50": {
                results.names[idx]: ap50_values[idx] for idx in range(num_classes)
            },
            "mAP50-95": {
                results.names[idx]: ap50_95_values[idx] for idx in range(num_classes)
            },
        }

        class_wise_results["precision"]["all"] = mp
        class_wise_results["recall"]["all"] = mr
        class_wise_results["mAP50"]["all"] = map50
        class_wise_results["mAP50-95"]["all"] = map5095

    elif task == "classify":
        class_wise_results = {
            "top1": {"all": results.top1},
            "top5": {"all": results.top5},
        }
    else:
        raise ValueError(f"Invalid task: {task}")

    # Convert to DataFrame
    table = pd.DataFrame(class_wise_results)
    table.index.name = "class"

    return table


def client_client_metrics(
    client_number,
    dataset_name,
    strategy_name,
    task,
    data_path,
    data_source_client=None,
    output_dirs=None,
):
    if output_dirs is None:
        output_dirs = get_output_dirs()

    # Use new output directory structure for loading client training logs
    logs_path = os.path.join(
        output_dirs["logs_clients"],
        f"client_{client_number}",
        f"training_{dataset_name}_{strategy_name}.log",
    )
    print(f"Loading logs from: {logs_path}")
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    print(f"Loading weights from: {weights}")
    model = load_fl_trained_model(weights, task, dataset_name, output_dirs)

    # Configure validation output to use new directory structure
    client_viz_dir = os.path.join(output_dirs["results_viz"], f"client_{client_number}")
    results = model.val(
        data=data_path,
        split="test",
        verbose=True,
        project=client_viz_dir,
        name=f"val_{dataset_name}",
    )
    table = get_classwise_results_table(results, task)

    # Create filename that distinguishes between model client and data source client
    if data_source_client is not None:
        filename = os.path.join(
            output_dirs["results_metrics"],
            f"client_{client_number}_on_client_{data_source_client}_data_{dataset_name}_{strategy_name}.csv",
        )
        description = (
            f"Client {client_number} model on Client {data_source_client} data"
        )
    else:
        filename = os.path.join(
            output_dirs["results_metrics"],
            f"client_{client_number}_own_data_{dataset_name}_{strategy_name}.csv",
        )
        description = f"Client {client_number} model on own data"

    safe_save_csv(table, filename, description)


def client_server_metrics(
    client_number, dataset_name, strategy_name, task, output_dirs=None
):
    if output_dirs is None:
        output_dirs = get_output_dirs()

    # Use new output directory structure for loading client training logs
    logs_path = os.path.join(
        output_dirs["logs_clients"],
        f"client_{client_number}",
        f"training_{dataset_name}_{strategy_name}.log",
    )
    print(f"Loading logs from: {logs_path}")
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    print(f"Loading weights from: {weights}")
    model = load_fl_trained_model(weights, task, dataset_name, output_dirs)

    # Configure validation output to use new directory structure
    client_viz_dir = os.path.join(output_dirs["results_viz"], f"client_{client_number}")
    results = model.val(
        data=f"{HOME}/datasets/{dataset_name}/data.yaml",
        split="test",
        verbose=True,
        project=client_viz_dir,
        name="val_server_data",
    )
    table = get_classwise_results_table(results, task)
    filename = os.path.join(
        output_dirs["results_metrics"],
        f"client_{client_number}_on_server_data_{dataset_name}_{strategy_name}.csv",
    )
    description = f"Client {client_number} model on server data"
    safe_save_csv(table, filename, description)


def server_client_metrics(
    client_number,
    dataset_name,
    strategy_name,
    num_rounds,
    task,
    data_path,
    output_dirs=None,
):
    if output_dirs is None:
        output_dirs = get_output_dirs()

    weights_path = os.path.join(
        output_dirs["checkpoints_server"],
        f"round_{num_rounds}_{dataset_name}_{strategy_name}.pt",
    )
    print(f"Loading server model weights from: {weights_path}")
    server_model = load_fl_trained_model(weights_path, task, dataset_name, output_dirs)

    if "head" in strategy_name.lower():
        base_weights_dir = output_dirs["weights_base"]
        server_model = load_head_only_weights(server_model, base_weights_dir)

    # Configure validation output to use new directory structure
    server_viz_dir = os.path.join(output_dirs["results_viz"], "server")
    results = server_model.val(
        data=data_path,
        split="test",
        verbose=True,
        project=server_viz_dir,
        name=f"val_client_{client_number}_data",
    )
    table = get_classwise_results_table(results, task)
    filename = os.path.join(
        output_dirs["results_metrics"],
        f"server_on_client_{client_number}_data_{dataset_name}_{strategy_name}.csv",
    )
    description = f"Server model on Client {client_number} data"
    safe_save_csv(table, filename, description)


def server_server_metrics(
    dataset_name, strategy_name, num_rounds, task, output_dirs=None
):
    if output_dirs is None:
        output_dirs = get_output_dirs()

    weights_path = os.path.join(
        output_dirs["checkpoints_server"],
        f"round_{num_rounds}_{dataset_name}_{strategy_name}.pt",
    )
    print(f"Loading server model weights from: {weights_path}")
    server_model = load_fl_trained_model(weights_path, task, dataset_name, output_dirs)

    if "head" in strategy_name.lower():
        base_weights_dir = output_dirs["weights_base"]
        server_model = load_head_only_weights(server_model, base_weights_dir)

    # Configure validation output to use new directory structure
    server_viz_dir = os.path.join(output_dirs["results_viz"], "server")
    results = server_model.val(
        data=f"{HOME}/datasets/{dataset_name}/data.yaml",
        split="test",
        verbose=True,
        project=server_viz_dir,
        name="val_server_data",
    )
    table = get_classwise_results_table(results, task)
    filename = os.path.join(
        output_dirs["results_metrics"],
        f"server_on_server_data_{dataset_name}_{strategy_name}.csv",
    )
    description = "Server model on server data"
    safe_save_csv(table, filename, description)


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="baseline")
    parser.add_argument("--strategy_name", type=str, default="FedAvg")
    parser.add_argument("--client_num", type=int, default=1)
    parser.add_argument("--scoring_style", type=str, default="client-client")
    parser.add_argument("--data_path", type=str)
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "segment", "pose", "classify"],
    )
    parser.add_argument(
        "--data_source_client",
        type=int,
        help="Client ID whose data is being used for evaluation",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    strategy_name = args.strategy_name
    client_num = args.client_num
    scoring_style = args.scoring_style
    num_rounds = SERVER_CONFIG["rounds"]
    data_path = args.data_path
    task = args.task
    data_source_client = args.data_source_client

    output_dirs = get_output_dirs()

    if scoring_style == "client-client":
        client_client_metrics(
            client_num,
            dataset_name,
            strategy_name,
            task,
            data_path,
            data_source_client,
            output_dirs,
        )
    elif scoring_style == "client-server":
        client_server_metrics(
            client_num, dataset_name, strategy_name, task, output_dirs
        )
    elif scoring_style == "server-client":
        server_client_metrics(
            client_num,
            dataset_name,
            strategy_name,
            num_rounds,
            task,
            data_path,
            output_dirs,
        )
    elif scoring_style == "server-server":
        server_server_metrics(
            dataset_name, strategy_name, num_rounds, task, output_dirs
        )
    else:
        raise ValueError(f"Invalid scoring_style: {scoring_style}")


if __name__ == "__main__":
    main()
