import pytest
from pathlib import Path
from collections import defaultdict
import builtins
import yaml
import shutil
import numpy as np
import os
from FedYOLO.data_partitioner.dirichlet_split import dirichlet_split_dataset  
import FedYOLO.data_partitioner.dirichlet_split as split_module

@pytest.fixture
def dummy_config(tmp_path):
    return {
        "dataset": str(tmp_path / "mock_dataset"),
        "num_clients": 2
    }

def test_dirichlet_split_dataset(monkeypatch, dummy_config):
    dummy_labels = {
        "train/labels/a.txt": [0],
        "train/labels/b.txt": [1],
        "valid/labels/c.txt": [0, 1],
        "test/labels/d.txt": [1],
    }

    dummy_path = Path(dummy_config["dataset"])
    written_yamls = {}

    #  Normalize path to dummy_labels key
    def relative_path_key(path):
        return str(path).replace(str(dummy_path) + os.sep, '').replace("\\", "/")

    #  Mock glob
    def mock_glob(self, pattern):
        split_name = self.parent.name
        return [dummy_path / k for k in dummy_labels if k.startswith(f"{split_name}/labels")]

    #  Mock label distribution function
    def mock_extract_label_distribution(path):
        key = relative_path_key(path)
        return defaultdict(int, {cls: 1 for cls in dummy_labels.get(key, [])})

    #  Mock file operations
    monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(Path, "glob", mock_glob)
    monkeypatch.setattr(split_module, "extract_label_distribution", mock_extract_label_distribution)

    #  Capture YAML writes
    def mock_open(filepath, mode='w'):
        class DummyFile:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def write(self, content): written_yamls[str(filepath)] = content
        return DummyFile()

    monkeypatch.setattr(builtins, "open", mock_open)
    monkeypatch.setattr(yaml, "dump", lambda data, f: f.write(str(data)))

    #  Run test
    dirichlet_split_dataset(dummy_config, alpha=0.5)

    #  Validate YAML contents
    assert any("client_0" in path for path in written_yamls)
    assert any("client_1" in path for path in written_yamls)

    for content in written_yamls.values():
        parsed = eval(content)
        assert parsed["train"] == "./train/images"
        assert parsed["val"] == "./valid/images"
        assert parsed["test"] == "./test/images"
        assert parsed["nc"] > 0
        assert isinstance(parsed["names"], list)
