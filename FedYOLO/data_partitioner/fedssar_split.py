import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from sklearn.cluster import OPTICS
from collections import defaultdict
from prettytable import PrettyTable

def extract_label_distribution(label_path):
    """Extract label distribution from a single label file."""
    class_counts = defaultdict(int)
    with open(label_path, 'r') as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            class_counts[class_id] += 1
    return class_counts

def normalize_distribution(distribution, class_ids):
    """Normalize the distribution vector based on available class ids."""
    vector = [distribution.get(cls_id, 0) for cls_id in class_ids]
    total = sum(vector)
    return [x / total if total > 0 else 0 for x in vector]

def fedssar_split_dataset(config):
    """
    FedSSaR-inspired data splitting using density-based OPTICS clustering.
    Saves client-specific datasets under datasets/bccd/partition_fedssar/
    """
    data_path = Path(config['dataset'])
    min_samples = config.get("min_samples", 5)
    num_clients = config['num_clients']

    splits = ['train', 'valid', 'test']
    label_files = []
    label_histograms = []

    print("[*] Extracting label distributions...")
    all_class_ids = set()
    for split in splits:
        split_path = data_path / split / 'labels'
        for file in split_path.glob('*.txt'):
            dist = extract_label_distribution(file)
            label_files.append((split, file))
            label_histograms.append(dist)
            all_class_ids.update(dist.keys())

    all_class_ids = sorted(list(all_class_ids))
    feature_vectors = np.array([
        normalize_distribution(d, all_class_ids) for d in label_histograms
    ])

    print("[*] Running OPTICS clustering...")
    clustering = OPTICS(min_samples=min_samples).fit(feature_vectors)
    cluster_labels = clustering.labels_

    # Group by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append((label_files[i][0], label_files[i][1]))

    print(f"[*] Found {len(clusters)} clusters")

    # Prepare partitions directory
    partition_path = data_path / "partition_fedssar"
    for i in range(num_clients):
        for split in splits:
            (partition_path / f"client_{i}" / split / "images").mkdir(parents=True, exist_ok=True)
            (partition_path / f"client_{i}" / split / "labels").mkdir(parents=True, exist_ok=True)

    # Assign clusters to clients in round-robin
    print("[*] Assigning clusters to clients...")
    cluster_items = list(clusters.items())
    cluster_items.sort(key=lambda x: len(x[1]), reverse=True)

    client_assignment = {f"client_{i}": [] for i in range(num_clients)}
    for idx, (_, items) in enumerate(cluster_items):
        client_id = f"client_{idx % num_clients}"
        client_assignment[client_id].extend(items)

    # Copy files to client folders
    for client_id, items in client_assignment.items():
        for split, label_file in items:
            image_file = str(label_file).replace('labels', 'images').replace('.txt', '.jpg')
            dest_image = partition_path / client_id / split / 'images' / Path(image_file).name
            dest_label = partition_path / client_id / split / 'labels' / label_file.name
            shutil.copy(image_file, dest_image)
            shutil.copy(label_file, dest_label)

        # Create client-specific data.yaml
        yaml_data = {
            'train': './train/images',
            'val': './valid/images',
            'test': './test/images',
            'nc': len(all_class_ids),
            'names': [str(i) for i in all_class_ids]
        }
        with open(partition_path / client_id / 'data.yaml', 'w') as f:
            yaml.dump(yaml_data, f)

    print(f"[*] Done. Data saved to: {partition_path}")

if __name__ == "__main__":
    SPLITS_CONFIG = {
        "dataset": "datasets/bccd",  # Adjust path if different
        "num_clients": 2,
        "min_samples": 3
    }
    fedssar_split_dataset(SPLITS_CONFIG)
