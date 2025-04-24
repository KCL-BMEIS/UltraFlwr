# config.py
import yaml

def get_nc_from_yaml(yaml_path):
    """Get number of classes from data.yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('nc', None)

def generate_client_config(num_clients, dataset_path, client_tasks):
    """Dynamically generate client configuration for n clients with specific tasks."""
    if len(client_tasks) != num_clients:
        raise ValueError("Length of client_tasks must match num_clients")
    return {
        i: {
            'cid': i,
            'data_path': f"{dataset_path}/partitions/client_{i}/data.yaml",
            'task': client_tasks[i]  # Assign task based on client index
        }
        for i in range(num_clients)
    }

# Base Configuration
BASE = "/home/localssk23"  # YOUR PATH CONTAINING UltraFlwr
HOME = f"{BASE}/UltraFlwr"

# Dataset Configuration
DATASET_NAME = 'baseline'
DATASET_NAME_SEG = 'baseline_seg'
DATASET_PATH = f'{HOME}/datasets/{DATASET_NAME}'
DATASET_PATH_SEG = f'{HOME}/datasets/{DATASET_NAME_SEG}'
DATA_YAML = f"{DATASET_PATH}/data.yaml"
DATA_YAML_SEG = f"{DATASET_PATH_SEG}/data.yaml"
NC = get_nc_from_yaml(DATA_YAML)
NC_SEG = get_nc_from_yaml(DATA_YAML_SEG)

# Number of clients can be easily modified here
NUM_CLIENTS = 2  # Change this to desired number of clients

# Manually assign data, task and dataset_name to each client
CLIENT_CONFIG = {
    0: {
        'cid': 0,
        'dataset_name': DATASET_NAME,
        'num_classes': NC,
        'data_path': f"{DATASET_PATH}/partitions/client_0/data.yaml",
        'task': 'detect',
    },
    1: {
        'cid': 1,
        'dataset_name': DATASET_NAME_SEG,
        'num_classes': NC_SEG,
        'data_path': f"{DATASET_PATH_SEG}/partitions/client_1/data.yaml",
        'task': 'segment',
    },
}

CLIENT_TASKS = [CLIENT_CONFIG[i]['task'] for i in range(NUM_CLIENTS)]
CLIENT_RATIOS = [1/NUM_CLIENTS] * NUM_CLIENTS

SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'num_clients': NUM_CLIENTS,
    'ratio': CLIENT_RATIOS
}

SERVER_CONFIG = {
    'server_address': "0.0.0.0:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': NUM_CLIENTS,
    'max_num_clients': NUM_CLIENTS * 2,  # Adjusted based on number of clients
    'strategy': 'FedNeckMedian',
}

YOLO_CONFIG = {
    'batch_size': 8,
    'epochs': 2,
}