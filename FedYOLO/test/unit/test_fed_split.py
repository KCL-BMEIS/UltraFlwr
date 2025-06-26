import pytest
from pathlib import Path
from collections import defaultdict
import yaml
import shutil

from FedYOLO.data_partitioner.dirichlet_split import dirichlet_split_dataset

# ---------- FIXTURE: Fake YOLO-style dataset ----------
@pytest.fixture
def fake_dirichlet_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    splits = ['train', 'valid', 'test']
    labels = {
        'train': [[0], [1], [0, 1]],
        'valid': [[0], [1]],
        'test': [[1]]
    }

    for split in splits:
        (dataset_dir / split / 'images').mkdir(parents=True)
        (dataset_dir / split / 'labels').mkdir(parents=True)

        for i, classes in enumerate(labels.get(split, [])):
            (dataset_dir / split / 'images' / f"{i}.jpg").touch()
            label_file = dataset_dir / split / 'labels' / f"{i}.txt"
            with open(label_file, 'w') as f:
                for cls in classes:
                    f.write(f"{cls} 0.5 0.5 1.0 1.0\n")

    return dataset_dir

# ---------- TEST: dirichlet_split_dataset ----------
def test_dirichlet_split_dataset(monkeypatch, fake_dirichlet_dataset):
    config = {
        'dataset': str(fake_dirichlet_dataset),
        'num_clients': 2
    }

    # Mock shutil.copy to avoid actual file copying
    monkeypatch.setattr(shutil, "copy", lambda src, dst: None)

    # Monkeypatch yaml.dump to skip writing actual content
    monkeypatch.setattr(yaml, "dump", lambda data, f: f.write(str(data)))

    # Run the function
    dirichlet_split_dataset(config, alpha=0.5)

    # Validate expected output structure
    partition_path = Path(config['dataset']) / "partition_dirichlet"
    for client_id in range(config['num_clients']):
        for split in ['train', 'valid', 'test']:
            img_dir = partition_path / f"client_{client_id}" / split / "images"
            lbl_dir = partition_path / f"client_{client_id}" / split / "labels"
            assert img_dir.exists(), f"{img_dir} not created"
            assert lbl_dir.exists(), f"{lbl_dir} not created"

        # Also check for data.yaml presence
        assert (partition_path / f"client_{client_id}" / "data.yaml").exists()
