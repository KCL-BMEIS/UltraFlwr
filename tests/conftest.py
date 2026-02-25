import pytest
import os
import yaml
import tempfile
from collections import OrderedDict
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_data_yaml(temp_dir):
    """Create a mock data.yaml file for testing."""
    yaml_path = os.path.join(temp_dir, "data.yaml")
    data = {
        "path": temp_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {0: "cat", 1: "dog", 2: "bird"},
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    return yaml_path


@pytest.fixture
def mock_state_dict():
    """Create a mock YOLO model state dict for testing strategies."""
    state_dict = OrderedDict()

    # Backbone layers (0-8)
    for i in range(9):
        state_dict[f"model.{i}.weight"] = torch.randn(10, 10)
        state_dict[f"model.{i}.bias"] = torch.randn(10)

    # Neck layers (9-22)
    for i in range(9, 23):
        state_dict[f"model.{i}.weight"] = torch.randn(10, 10)
        state_dict[f"model.{i}.bias"] = torch.randn(10)

    # Head layer (23)
    state_dict["model.23.weight"] = torch.randn(10, 10)
    state_dict["model.23.bias"] = torch.randn(10)

    return state_dict
