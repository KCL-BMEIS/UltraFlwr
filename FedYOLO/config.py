# config.py
import yaml

def get_nc_from_yaml(yaml_path):
    """Get number of classes from data.yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('nc', None)

def generate_client_config(num_clients, dataset_path):
    """Dynamically generate client configuration for n clients."""
    return {
        i: {
            'cid': i,
            'data_path': f"{dataset_path}/partitions/client_{i}/data.yaml"
        }
        for i in range(num_clients)
    }

# Base Configuration
BASE = "/home/yang/Downloads/upgrade_flower"  # YOUR PATH CONTAINING UltraFlwr
HOME = f"{BASE}/UltraFlwr"
DATASET_NAME = 'baseline'
DATASET_PATH = f'{HOME}/datasets/{DATASET_NAME}'
DATA_YAML = f"{DATASET_PATH}/data.yaml"
NC = get_nc_from_yaml(DATA_YAML)

# Number of clients can be easily modified here
NUM_CLIENTS = 2  # Change this to desired number of clients

# Generate equal ratios for n clients
CLIENT_RATIOS = [1/NUM_CLIENTS] * NUM_CLIENTS

SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'num_clients': NUM_CLIENTS,
    'ratio': CLIENT_RATIOS
}

# Dynamically generate client config
CLIENT_CONFIG = generate_client_config(NUM_CLIENTS, DATASET_PATH)

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
    'epochs': 1,
}