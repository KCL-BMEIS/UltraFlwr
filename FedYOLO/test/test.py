from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path
import os # Import os module

# Import CLIENT_CONFIG
from FedYOLO.config import HOME, SERVER_CONFIG, CLIENT_CONFIG

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='baseline')
parser.add_argument('--strategy_name', type=str, default='FedAvg')
parser.add_argument('--client_num', type=int, default=1)
parser.add_argument('--scoring_style', type=str, default="client-client")

args = parser.parse_args()

dataset_name = args.dataset_name
strategy_name = args.strategy_name
client_num = args.client_num
scoring_style = args.scoring_style
num_rounds = SERVER_CONFIG['rounds']


def get_classwise_results_table(results):
    # Access precision, recall, and mAP values directly as arrays
    precision_values = results.box.p  # List/array of precision values for each class
    recall_values = results.box.r  # List/array of recall values for each class
    ap50_values = results.box.ap50  # Array of AP50 values for each class
    ap50_95_values = results.box.ap  # Array of AP50-95 values for each class

    # Ensure alignment between metrics and class names
    num_classes = min(len(results.names), len(precision_values))

    # Construct class-wise results table
    class_wise_results = {
        'precision': {results.names[idx]: precision_values[idx] for idx in range(num_classes)},
        'recall': {results.names[idx]: recall_values[idx] for idx in range(num_classes)},
        'mAP50': {results.names[idx]: ap50_values[idx] for idx in range(num_classes)},
        'mAP50-95': {results.names[idx]: ap50_95_values[idx] for idx in range(num_classes)}
    }

    # Calculate mean results (overall "all" row)
    mp, mr, map50, map5095 = results.box.mean_results()
    class_wise_results['precision']['all'] = mp
    class_wise_results['recall']['all'] = mr
    class_wise_results['mAP50']['all'] = map50
    class_wise_results['mAP50-95']['all'] = map5095

    # Convert to DataFrame
    table = pd.DataFrame(class_wise_results)
    table.index.name = 'class'

    return table


def client_client_metrics(client_number, dataset_name, strategy_name):
    # Get client-specific configuration
    client_cfg = CLIENT_CONFIG[client_number]
    client_dataset_name = client_cfg['dataset_name']
    # Construct the log path using the client's specific dataset name
    logs_path = f"{HOME}/logs/client_{client_number}_log_{client_dataset_name}_{strategy_name}.txt"
    try:
        weights_path = extract_results_path(logs_path)
        weights = f"{HOME}/{weights_path}/weights/best.pt"
        if not os.path.exists(weights):
             print(f"Weight file not found: {weights}. Skipping client-client test for client {client_number}, strategy {strategy_name}.")
             return None
        model = YOLO(weights)
        # Use the client's specific data path for validation
        results = model.val(data=f'{HOME}/datasets/{client_dataset_name}/partitions/client_{client_number}/data.yaml', split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with client-specific dataset name in the filename
        output_path = f"{HOME}/results/client_{client_number}_results_{client_dataset_name}_{strategy_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Client-client results saved to {output_path}")
    except FileNotFoundError:
        print(f"Log file not found: {logs_path}. Skipping client-client test for client {client_number}, strategy {strategy_name}.")
        return None
    except Exception as e:
        print(f"An error occurred during client-client test for client {client_number}, strategy {strategy_name}: {e}")
        return None


def client_server_metrics(client_number, dataset_name, strategy_name):
    # Get client-specific configuration for log path and task
    client_cfg = CLIENT_CONFIG[client_number]
    client_dataset_name = client_cfg['dataset_name']
    client_task = client_cfg['task']

    # --- Task Mismatch Check ---
    # Assuming 'baseline' dataset implies 'detect' task for the server's global test set
    if client_task == 'segment' and dataset_name == 'baseline':
        print(f"Skipping client-server test for client {client_number} (segment task) on dataset '{dataset_name}' (detect task). Incompatible.")
        return None
    # Add similar checks if other base datasets imply different tasks (e.g., if dataset_name == 'baseline_seg' implies 'segment')
    # if client_task == 'detect' and dataset_name == 'baseline_seg':
    #     print(f"Skipping client-server test for client {client_number} (detect task) on dataset '{dataset_name}' (segment task). Incompatible.")
    #     return None
    # --- End Task Mismatch Check ---

    # Construct the log path using the client's specific dataset name
    logs_path = f"{HOME}/logs/client_{client_number}_log_{client_dataset_name}_{strategy_name}.txt"
    try:
        weights_path = extract_results_path(logs_path)
        weights = f"{HOME}/{weights_path}/weights/best.pt"
        if not os.path.exists(weights):
             print(f"Weight file not found: {weights}. Skipping client-server test for client {client_number}, strategy {strategy_name}.")
             return None
        model = YOLO(weights)
        # Use the main dataset's test set (passed as dataset_name) for validation
        results = model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with client-specific dataset name and "_server" suffix in the filename
        output_path = f"{HOME}/results/client_{client_number}_results_{client_dataset_name}_{strategy_name}_server.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Client-server results saved to {output_path}")
    except FileNotFoundError:
        print(f"Log file not found: {logs_path}. Skipping client-server test for client {client_number}, strategy {strategy_name}.")
        return None
    except IndexError as e:
         print(f"IndexError during client-server validation for client {client_number}, strategy {strategy_name} on dataset {dataset_name}: {e}. Likely task mismatch.")
         return None
    except Exception as e:
        print(f"An error occurred during client-server test for client {client_number}, strategy {strategy_name}: {e}")
        return None


def server_client_metrics(client_number, dataset_name, strategy_name, num_rounds):
    # Get client-specific configuration for data path
    client_cfg = CLIENT_CONFIG[client_number]
    client_dataset_name = client_cfg['dataset_name']
    client_data_path = client_cfg['data_path'] # Use the full path from config

    # Server weights path uses the base dataset_name
    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"

    if not os.path.exists(weights_path):
        print(f"Server weight file not found: {weights_path}. Skipping server-client test for client {client_number}, strategy {strategy_name}.")
        return None

    try:
        server_model = YOLO(weights_path)
        # Use the client's specific data path for validation
        results = server_model.val(data=client_data_path, split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with client-specific dataset name in the filename
        output_path = f"{HOME}/results/server_client_{client_number}_results_{client_dataset_name}_{strategy_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Server-client results saved to {output_path}")
    except IndexError as e:
         print(f"IndexError during server-client validation for client {client_number}, strategy {strategy_name}: {e}. Likely task mismatch between server model and client data.")
         return None
    except Exception as e:
        print(f"An error occurred during server-client test for client {client_number}, strategy {strategy_name}: {e}")
        return None


def server_server_metrics(dataset_name, strategy_name, num_rounds):
    # Server weights path uses the base dataset_name
    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"

    if not os.path.exists(weights_path):
        print(f"Server weight file not found: {weights_path}. Skipping server-server test for strategy {strategy_name}.")
        return None

    try:
        server_model = YOLO(weights_path)
        # Use the main dataset's test set (passed as dataset_name) for validation
        results = server_model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with base dataset name
        output_path = f"{HOME}/results/server_results_{dataset_name}_{strategy_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Server-server results saved to {output_path}")
    except IndexError as e:
         print(f"IndexError during server-server validation for strategy {strategy_name} on dataset {dataset_name}: {e}. Likely task mismatch.")
         return None
    except Exception as e:
        print(f"An error occurred during server-server test for strategy {strategy_name}: {e}")
        return None

if scoring_style == "client-client":
    client_client_metrics(client_num, dataset_name, strategy_name) # Removed assignment as functions now save directly
elif scoring_style == "client-server":
    client_server_metrics(client_num, dataset_name, strategy_name)
elif scoring_style == "server-client":
    server_client_metrics(client_num, dataset_name, strategy_name, num_rounds)
elif scoring_style == "server-server":
    server_server_metrics(dataset_name, strategy_name, num_rounds)
else:
    raise ValueError(f"Invalid scoring_style: {scoring_style}")
