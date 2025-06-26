import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict


def extract_label_distribution(label_path):
    class_counts = defaultdict(int)
    with open(label_path, 'r') as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            class_counts[class_id] += 1
    return class_counts


def dirichlet_split_dataset(config, alpha=0.5):
    data_path = Path(config['dataset'])
    num_clients = config['num_clients']
    splits = ['train', 'valid', 'test']
    all_class_ids = set()
    file_label_map = []

    print("[*] Reading label files...")
    for split in splits:
        split_path = data_path / split / 'labels'
        for file in split_path.glob('*.txt'):
            dist = extract_label_distribution(file)
            if dist:
                file_label_map.append((split, file, list(dist.keys())))
                all_class_ids.update(dist.keys())

    all_class_ids = sorted(list(all_class_ids))
    class_to_files = defaultdict(list)
    for split, file, classes in file_label_map:
        for cls in classes:
            class_to_files[cls].append((split, file))

    print(f"[*] Splitting using Dirichlet distribution (alpha={alpha})...")
    client_data = defaultdict(list)
    for cls in all_class_ids:
        files = class_to_files[cls]
        if not files:
            continue
        probs = np.random.dirichlet([alpha] * num_clients)
        splits_per_client = np.random.multinomial(len(files), probs)
        idx = 0
        for client_id, count in enumerate(splits_per_client):
            for _ in range(count):
                if idx < len(files):
                    client_data[client_id].append(files[idx])
                    idx += 1

    # Create client folders and copy files
    partition_path = data_path / "partition_dirichlet"
    for client_id in range(num_clients):
        for split in splits:
            (partition_path / f"client_{client_id}" / split / "images").mkdir(parents=True, exist_ok=True)
            (partition_path / f"client_{client_id}" / split / "labels").mkdir(parents=True, exist_ok=True)

        for split, label_file in client_data[client_id]:
            image_file = str(label_file).replace('labels', 'images').replace('.txt', '.jpg')
            dest_image = partition_path / f"client_{client_id}" / split / 'images' / Path(image_file).name
            dest_label = partition_path / f"client_{client_id}" / split / 'labels' / label_file.name
            shutil.copy(image_file, dest_image)
            shutil.copy(label_file, dest_label)

        yaml_data = {
            'train': './train/images',
            'val': './valid/images',
            'test': './test/images',
            'nc': len(all_class_ids),
            'names': [str(i) for i in all_class_ids]
        }
        with open(partition_path / f"client_{client_id}" / 'data.yaml', 'w') as f:
            yaml.dump(yaml_data, f)

    print(f"[*] Done. Data saved to: {partition_path}")


if __name__ == "__main__":
    SPLITS_CONFIG = {
        "dataset": "datasets/bccd",  # Change if needed
        "num_clients": 3
    }
    dirichlet_split_dataset(SPLITS_CONFIG, alpha=0.5)
