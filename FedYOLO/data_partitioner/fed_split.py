import yaml
import shutil
from pathlib import Path
from prettytable import PrettyTable
from FedYOLO.config import HOME, CLIENT_CONFIG, CLIENT_RATIOS, NUM_CLIENTS

def count_classes(label_files):
    """Count occurrences of each class in the label files."""
    class_counts = {}
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts

def create_class_distribution_table(global_counts, client_counts, split):
    """Create a table comparing global and client-specific class distributions."""
    classes = sorted(set(global_counts.get(split, {}).keys()).union(
        *[client_counts[client].get(split, {}).keys() for client in client_counts]
    ))
    
    table = PrettyTable()
    table.field_names = ["Class", "Global Count"] + [f"{client} Count" for client in client_counts]
    
    for class_id in classes:
        row = [
            class_id, 
            global_counts.get(split, {}).get(class_id, 0), 
            *[client_counts[client].get(split, {}).get(class_id, 0) for client in client_counts]
        ]
        table.add_row(row)
    
    return table

def split_dataset(config):
    """
    Split dataset for federated learning with n clients
    Args:
        config (dict): Configuration dictionary containing:
            - ratio (list): List of ratios for each client
            - dataset (str): Path to dataset directory
            - num_clients (int): Number of clients
            - dataset_name (str): Name of the dataset
    """
    ratios = config['ratio']
    data_path = Path(config['dataset'])
    num_clients = config['num_clients']
    dataset_name = config['dataset_name']

    print(f"\nProcessing dataset: {dataset_name}")

    # Validate inputs
    if not isinstance(ratios, list) or len(ratios) != num_clients:
        raise ValueError(f"Ratios list must have length {num_clients}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Read original yaml file
    with open(data_path / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    # Count global class-wise information
    global_class_counts = {'train': {}, 'valid': {}, 'test': {}, 'total': {}}
    for split in ['train', 'valid', 'test']:
        label_files = list((data_path / split / 'labels').glob('*'))
        split_class_counts = count_classes(label_files)
        for class_id, count in split_class_counts.items():
            global_class_counts[split][class_id] = count
            global_class_counts['total'][class_id] = global_class_counts['total'].get(class_id, 0) + count

    partition_path = data_path / 'partitions'
    
    # Find which clients use this dataset
    clients_for_dataset = [cid for cid, client_data in CLIENT_CONFIG.items() 
                           if client_data['dataset_name'] == dataset_name]
    
    if not clients_for_dataset:
        print(f"No clients are using dataset {dataset_name}, skipping split creation")
        return
    
    print(f"Creating splits for clients: {clients_for_dataset}")
    
    # Create client directories and yaml files for clients using this dataset
    for client_id in clients_for_dataset:
        client_dir = partition_path / f'client_{client_id}'
        for split in ['train', 'valid', 'test']:
            (client_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (client_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Create client yaml
        client_yaml = data.copy()
        client_yaml['train'] = './train/images'
        client_yaml['val'] = './valid/images'
        client_yaml['test'] = './test/images'

        with open(client_dir / 'data.yaml', 'w') as f:
            yaml.dump(client_yaml, f)

    client_class_counts = {f'client_{i}': {'train': {}, 'valid': {}, 'test': {}} 
                          for i in clients_for_dataset}

    # Split and copy files
    for split in ['train', 'valid', 'test']:
        images = list((data_path / split / 'images').glob('*'))
        labels = list((data_path / split / 'labels').glob('*'))
        
        if not images or not labels:
            print(f"No {split} data found for {dataset_name}")
            continue
            
        # Adjust ratios for the subset of clients using this dataset
        num_dataset_clients = len(clients_for_dataset)
        dataset_ratios = [1/num_dataset_clients] * num_dataset_clients
        
        start_idx = 0
        remaining = len(images)

        for i, client_id in enumerate(clients_for_dataset):
            # For last client, use all remaining files
            if i == num_dataset_clients - 1:
                n_files = remaining
            else:
                n_files = int(len(images) * dataset_ratios[i])
                remaining -= n_files
            
            client_images = images[start_idx:start_idx + n_files]
            client_labels = labels[start_idx:start_idx + n_files]
            
            client_dir = partition_path / f'client_{client_id}'
            for img in client_images:
                shutil.copy2(img, client_dir / split / 'images')
            for lbl in client_labels:
                shutil.copy2(lbl, client_dir / split / 'labels')
            
            start_idx += n_files
            
            # Count class-wise information
            class_counts = count_classes(client_labels)
            client_class_counts[f'client_{client_id}'][split] = class_counts

        # Print the table for this split
        table = create_class_distribution_table(global_class_counts, client_class_counts, split)
        print(f"\nClass distribution for {dataset_name} {split} split:")
        print(table)

def generate_splits_configs():
    """Generate split configurations for each unique dataset in CLIENT_CONFIG."""
    datasets = {}
    
    # Collect unique datasets
    for client_data in CLIENT_CONFIG.values():
        dataset_name = client_data['dataset_name']
        if dataset_name not in datasets:
            dataset_path = f'{HOME}/datasets/{dataset_name}'
            datasets[dataset_name] = {
                'dataset_name': dataset_name,
                'dataset': dataset_path,
                'num_clients': NUM_CLIENTS,
                'ratio': CLIENT_RATIOS
            }
    
    return datasets

if __name__ == "__main__":
    # Generate split configs for each dataset
    splits_configs = generate_splits_configs()
    
    # Process each dataset
    for dataset_name, config in splits_configs.items():
        split_dataset(config)