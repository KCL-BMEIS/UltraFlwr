from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path

from FedYOLO.config import HOME, SERVER_CONFIG

import pandas as pd
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='baseline')
parser.add_argument('--strategy_name', type=str, default='FedAvg')
parser.add_argument('--client_num', type=int, default=1)
parser.add_argument('--scoring_style', type=str, default="client-client")
parser.add_argument('--data_path', type=str)
parser.add_argument('--task', type=str, default="detect", choices=["detect", "segment", "pose", "classify"], help="Task type: 'detect' for detection, 'segment' for segmentation, 'pose' for pose estimation, 'classify' for classification")
parser.add_argument('--data_source_client', type=int, help="Client ID whose data is being used for evaluation (for cross-client testing)")

args = parser.parse_args()

dataset_name = args.dataset_name
strategy_name = args.strategy_name
client_num = args.client_num
scoring_style = args.scoring_style
num_rounds = SERVER_CONFIG['rounds']
data_path = args.data_path
task = args.task
data_source_client = args.data_source_client


def safe_save_csv(table, filename, description=""):
    """Safely save CSV file with error handling and fallback location"""
    try:
        # Ensure results directory exists
        results_dir = os.path.dirname(filename)
        os.makedirs(results_dir, exist_ok=True)
        
        # Try to save the CSV
        table.to_csv(filename, index=True, index_label='class')
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
        table.to_csv(fallback_filename, index=True, index_label='class')
        print(f"⚠ Permission denied for {filename}")
        print(f"✓ Saved {description} to fallback location: {fallback_filename}")
        
    except Exception as e:
        print(f"✗ Error saving {description} to {filename}: {str(e)}")


def list_csv_files():
    """List all CSV files in the results directory for debugging"""
    results_dir = f"{HOME}/results"
    print(f"\n=== CSV files in {results_dir} ===")
    try:
        if os.path.exists(results_dir):
            csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
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


def get_classwise_results_table(results, task):
    if task == 'detect':
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
        
    elif task == 'segment':
        # Access precision, recall, and mAP values directly as arrays
        precision_values = results.seg.p
        recall_values = results.seg.r
        ap50_values = results.seg.ap50
        ap50_95_values = results.seg.ap
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
        mp, mr, map50, map5095 = results.seg.mean_results()

    elif task == 'pose':
        # Access pose-specific evaluation metrics
        precision_values = results.pose.p
        recall_values = results.pose.r
        ap50_values = results.pose.ap50
        ap50_95_values = results.pose.ap
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
        mp, mr, map50, map5095 = results.pose.mean_results()

    elif task == 'classify':
        # Access classification-specific evaluation metrics
        top1_value = results.top1 
        top5_value = results.top5 

        class_wise_results = {
            'top1': {},
            'top5': {}
        }

    else:
        raise ValueError(f"Invalid task: {task}")

    if task in ['detect', 'segment', 'pose']:
        class_wise_results['precision']['all'] = mp
        class_wise_results['recall']['all'] = mr
        class_wise_results['mAP50']['all'] = map50
        class_wise_results['mAP50-95']['all'] = map5095
    elif task == 'classify':
        # Add mean results for classification
        class_wise_results['top1']['all'] = top1_value
        class_wise_results['top5']['all'] = top5_value

    # Convert to DataFrame
    table = pd.DataFrame(class_wise_results)
    table.index.name = 'class'

    return table


def client_client_metrics(client_number, dataset_name, strategy_name, task, data_path, data_source_client=None):

    logs_path = f"{HOME}/logs/client_{client_number}_log_{dataset_name}_{strategy_name}.txt"
    print(f"Loading logs from: {logs_path}")
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    print(f"Loading weights from: {weights}")
    model = YOLO(weights)
    results = model.val(data=data_path, split="test", verbose=True)
    table = get_classwise_results_table(results, task)
    
    # Create filename that distinguishes between model client and data source client
    if data_source_client is not None:
        filename = f"{HOME}/results/client_{client_number}_on_client_{data_source_client}_data_results_{dataset_name}_{strategy_name}.csv"
        description = f"Client {client_number} model on Client {data_source_client} data"
    else:
        filename = f"{HOME}/results/client_{client_number}_results_{dataset_name}_{strategy_name}.csv"
        description = f"Client {client_number} model on own data"
    
    safe_save_csv(table, filename, description)
    list_csv_files()

def client_server_metrics(client_number, dataset_name, strategy_name, task):

    logs_path = f"{HOME}/logs/client_{client_number}_log_{dataset_name}_{strategy_name}.txt"
    print(f"Loading logs from: {logs_path}")
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    print(f"Loading weights from: {weights}")
    model = YOLO(weights)
    results = model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', split="test", verbose=True)
    table = get_classwise_results_table(results, task)
    filename = f"{HOME}/results/client_{client_number}_results_{dataset_name}_{strategy_name}_server.csv"
    description = f"Client {client_number} model on server data"
    safe_save_csv(table, filename, description)
    list_csv_files()

def server_client_metrics(client_number, dataset_name, strategy_name, num_rounds, task, data_path):

    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"
    print(f"Loading server model weights from: {weights_path}")
    server_model = YOLO(weights_path)
    normal_model = YOLO()

    if 'head' in strategy_name.lower():
        detection_weights = {k: v for k, v in server_model.model.state_dict().items() if k.startswith('model.detect')}
        normal_model.model.load_state_dict({**normal_model.model.state_dict(), **detection_weights}, strict=False)   
        server_model = normal_model 
    
    results = server_model.val(data=data_path, split="test", verbose=True)
    table = get_classwise_results_table(results, task)
    filename = f"{HOME}/results/server_on_client_{client_number}_data_results_{dataset_name}_{strategy_name}.csv"
    description = f"Server model on Client {client_number} data"
    safe_save_csv(table, filename, description)
    list_csv_files()

def server_server_metrics(dataset_name, strategy_name, num_rounds, task):

    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"
    print(f"Loading server model weights from: {weights_path}")
    server_model = YOLO(weights_path)
    normal_model = YOLO()

    if 'head' in strategy_name.lower():
        detection_weights = {k: v for k, v in server_model.model.state_dict().items() if k.startswith('model.detect')}
        normal_model.model.load_state_dict({**normal_model.model.state_dict(), **detection_weights}, strict=False)   
        server_model = normal_model 
    
    results = server_model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', split="test", verbose=True)
    table = get_classwise_results_table(results, task)
    filename = f"{HOME}/results/server_results_{dataset_name}_{strategy_name}.csv"
    description = f"Server model on server data"
    safe_save_csv(table, filename, description)
    list_csv_files()

if scoring_style == "client-client":
    client_metrics_table = client_client_metrics(client_num, dataset_name, strategy_name, task, data_path, data_source_client)
elif scoring_style == "client-server":
    client_metrics_table = client_server_metrics(client_num, dataset_name, strategy_name, task)
elif scoring_style == "server-client":
    client_metrics_table = server_client_metrics(client_num, dataset_name, strategy_name, num_rounds, task, data_path)
elif scoring_style == "server-server":
    client_metrics_table = server_server_metrics(dataset_name, strategy_name, num_rounds, task)
else:
    raise ValueError(f"Invalid scoring_style: {scoring_style}")
