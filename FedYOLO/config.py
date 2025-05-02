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
BASE = "/home/yang/Downloads/upgrade_flower"
HOME = f"{BASE}/UltraFlwr"

# --- Multi-client, multi-task configuration ---

# List of detection and segmentation datasets
DETECTION_DATASETS = ['surg_od']
SEGMENTATION_DATASETS = ['Endonet_seg']

# --- Generalized approach: specify number of clients per dataset as a dictionary ---
DETECTION_CLIENTS = {'surg_od': 1}         # dataset_name: num_clients
SEGMENTATION_CLIENTS = {'Endonet_seg': 1}  # dataset_name: num_clients

client_specs = []
for ds, n_clients in DETECTION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({'dataset_name': ds, 'task': 'detect', 'client_idx': i})
for ds, n_clients in SEGMENTATION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({'dataset_name': ds, 'task': 'segment', 'client_idx': i})

NUM_CLIENTS = len(client_specs)

# Build CLIENT_CONFIG
CLIENT_CONFIG = {}
for cid, spec in enumerate(client_specs):
    dataset_name = spec['dataset_name']
    task = spec['task']
    client_idx = spec['client_idx']
    dataset_path = f"{HOME}/datasets/{dataset_name}"
    data_yaml = f"{dataset_path}/data.yaml"
    nc = get_nc_from_yaml(data_yaml)
    data_path = f"{dataset_path}/partitions/client_{client_idx}/data.yaml"
    CLIENT_CONFIG[cid] = {
        'cid': cid,
        'dataset_name': dataset_name,
        'num_classes': nc,
        'data_path': data_path,
        'task': task,
    }

CLIENT_TASKS = [CLIENT_CONFIG[i]['task'] for i in range(NUM_CLIENTS)]
CLIENT_RATIOS = [1/NUM_CLIENTS] * NUM_CLIENTS

# For backward compatibility, set the first detection and segmentation dataset names
DATASET_NAME = DETECTION_DATASETS[0] if DETECTION_DATASETS else ''
DATASET_NAME_SEG = SEGMENTATION_DATASETS[0] if SEGMENTATION_DATASETS else ''
DATASET_PATH = f'{HOME}/datasets/{DATASET_NAME}'
DATASET_PATH_SEG = f'{HOME}/datasets/{DATASET_NAME_SEG}'
DATA_YAML = f"{DATASET_PATH}/data.yaml"
DATA_YAML_SEG = f"{DATASET_PATH_SEG}/data.yaml"
NC = get_nc_from_yaml(DATA_YAML) if DATASET_NAME else None
NC_SEG = get_nc_from_yaml(DATA_YAML_SEG) if DATASET_NAME_SEG else None

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
    'batch_size': 2,
    'epochs': 2,
}