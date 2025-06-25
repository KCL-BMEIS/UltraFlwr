# config.py

import yaml
import os

def get_nc_from_yaml(yaml_path):
    """Get number of classes from data.yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('nc', None)

def generate_client_config(num_clients, partition_path):
    """Dynamically generate client configuration for n clients."""
    return {
        i: {
            'cid': i,
            'data_path': os.path.join(partition_path, f"client_{i}", "data.yaml")
        }
        for i in range(num_clients)
    }

# --- Base Paths ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = BASE

# --- Dataset & Split Configuration ---
DATASET_NAME = 'bccd'
PARTITION_METHOD = 'dirichlet'  # ✅ CHANGED TO DIRICHLET
PARTITION_TYPE = f'partition_{PARTITION_METHOD}'  # e.g., partition_dirichlet
DATASET_PATH = os.path.join(HOME, 'datasets', DATASET_NAME)
PARTITION_PATH = os.path.join(DATASET_PATH, PARTITION_TYPE)

# --- Dataset Info ---
DATA_YAML = os.path.join(DATASET_PATH, 'data.yaml')
NC = get_nc_from_yaml(DATA_YAML)

# --- Client Config ---
NUM_CLIENTS = 2
CLIENT_RATIOS = [1 / NUM_CLIENTS] * NUM_CLIENTS
CLIENT_CONFIG = generate_client_config(NUM_CLIENTS, PARTITION_PATH)

# --- Fed Splitting Config ---
SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'partition_method': PARTITION_METHOD,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'partition_path': PARTITION_PATH,
    'num_clients': NUM_CLIENTS,
    'ratio': CLIENT_RATIOS,
    'min_samples': 3,
}

# --- Server Config ---
SERVER_CONFIG = {
    'server_address': "127.0.0.1:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': NUM_CLIENTS,
    'max_num_clients': NUM_CLIENTS * 2,
    'strategy': 'FedAvg',
}

# --- YOLO Training Config ---
YOLO_CONFIG = {
    'batch_size': 8,
    'epochs': 1,
}

# --- Logs Path ---
LOGS_DIR = os.path.join(HOME, 'logs', f"{SERVER_CONFIG['strategy']}_{DATASET_NAME}_{PARTITION_METHOD}")
os.makedirs(LOGS_DIR, exist_ok=True)
