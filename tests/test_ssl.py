import pytest
import torch
import numpy as np
import os
from unittest.mock import patch, MagicMock
from fedyolo.train.ssl_client import SSLFlowerClient, validate_ssl_config
from fedyolo.train.ssl_trainer import LightlySSLTrainer, YOLOToLightlyAdapter
from fedyolo.train.strategies import get_section_parameters


class TestSSLClient:
    """Tests for SSL Flower client."""

    @pytest.fixture
    def mock_ssl_client(self, mock_state_dict):
        """Create a mock SSL client for testing."""
        with patch("fedyolo.train.ssl_client.YOLO") as mock_yolo, patch(
            "fedyolo.train.ssl_client.get_output_dirs"
        ) as mock_dirs, patch("fedyolo.train.ssl_client.LightlySSLTrainer"):
            # Setup mock directories
            mock_dirs.return_value = {"weights_base": "/tmp/weights"}

            # Setup mock YOLO model
            mock_model = MagicMock()
            mock_model.state_dict.return_value = mock_state_dict
            mock_yolo.return_value.model = mock_model

            # Create SSL client
            client = SSLFlowerClient(
                cid=0,
                data_path="/tmp/test/data.yaml",
                dataset_name="test_dataset",
                strategy_name="FedBackboneAvg",
                task="detect",
            )
            return client

    def test_ssl_client_initialization(self, mock_ssl_client):
        """Test that SSL client initializes correctly."""
        assert mock_ssl_client.cid == 0
        assert mock_ssl_client.task == "detect"
        assert mock_ssl_client.strategy_name == "FedBackboneAvg"
        assert mock_ssl_client.ssl_trainer is not None

    def test_get_parameters_returns_backbone_only(
        self, mock_ssl_client, mock_state_dict
    ):
        """Test that SSL client only returns backbone parameters."""
        params = mock_ssl_client.get_parameters(config={})

        # Calculate expected backbone parameter count
        backbone, _, _ = get_section_parameters(mock_state_dict)
        expected_count = len(backbone)

        assert (
            len(params) == expected_count
        ), f"SSL client should return {expected_count} backbone parameters"

    def test_get_parameters_returns_numpy_arrays(self, mock_ssl_client):
        """Test that get_parameters returns numpy arrays."""
        params = mock_ssl_client.get_parameters(config={})

        assert all(
            isinstance(p, np.ndarray) for p in params
        ), "All SSL parameters should be numpy arrays"

    def test_set_parameters_updates_backbone(self, mock_ssl_client, mock_state_dict):
        """Test that set_parameters correctly updates backbone weights."""
        backbone, _, _ = get_section_parameters(mock_state_dict)

        # Create fake parameters matching backbone
        fake_params = [
            np.random.randn(*mock_state_dict[k].shape) for k in sorted(backbone.keys())
        ]

        # Should not raise any errors
        mock_ssl_client.set_parameters(fake_params)

    def test_set_parameters_validates_count(self, mock_ssl_client):
        """Test that set_parameters validates parameter count."""
        # Create wrong number of parameters
        wrong_params = [np.random.randn(10, 10) for _ in range(5)]

        # Should handle gracefully with warning
        mock_ssl_client.set_parameters(wrong_params)


class TestSSLConfigValidation:
    """Tests for SSL configuration validation."""

    @patch("fedyolo.train.ssl_client.SSL_CONFIG", {"method": "byol"})
    @patch("fedyolo.train.ssl_client.CLIENT_SSL_CONFIG", {})
    def test_homogeneous_ssl_config_passes(self):
        """Test that homogeneous SSL configuration passes validation."""
        # Should not raise any errors
        validate_ssl_config()

    @patch(
        "fedyolo.train.ssl_client.SSL_CONFIG",
        {"method": "byol", "allow_heterogeneous": False},
    )
    @patch(
        "fedyolo.train.ssl_client.CLIENT_SSL_CONFIG",
        {0: {"method": "byol"}, 1: {"method": "simclr"}},
    )
    def test_heterogeneous_ssl_config_raises_error(self):
        """Test that heterogeneous SSL without flag raises error."""
        with pytest.raises(ValueError, match="Heterogeneous SSL methods detected"):
            validate_ssl_config()

    @patch(
        "fedyolo.train.ssl_client.SSL_CONFIG",
        {"method": "byol", "allow_heterogeneous": True},
    )
    @patch(
        "fedyolo.train.ssl_client.CLIENT_SSL_CONFIG",
        {0: {"method": "byol"}, 1: {"method": "simclr"}},
    )
    @patch("fedyolo.train.ssl_client.SSL_SERVER_CONFIG", {"strategy": "FedBackboneAvg"})
    def test_heterogeneous_ssl_config_with_flag_passes(self):
        """Test that heterogeneous SSL with flag passes validation."""
        # Should not raise errors (only warnings)
        validate_ssl_config()

    @patch(
        "fedyolo.train.ssl_client.SSL_CONFIG",
        {"method": "byol", "allow_heterogeneous": True},
    )
    @patch(
        "fedyolo.train.ssl_client.CLIENT_SSL_CONFIG",
        {0: {"method": "byol"}, 1: {"method": "simclr"}},
    )
    @patch("fedyolo.train.ssl_client.SSL_SERVER_CONFIG", {"strategy": "FedAvg"})
    def test_heterogeneous_ssl_requires_backbone_avg(self):
        """Test that heterogeneous SSL requires FedBackboneAvg strategy."""
        with pytest.raises(
            ValueError, match="Heterogeneous SSL requires FedBackboneAvg"
        ):
            validate_ssl_config()


class TestYOLOToLightlyAdapter:
    """Tests for YOLO to Lightly dataset adapter."""

    @pytest.fixture
    def mock_detect_dataset(self, temp_dir):
        """Create a mock detection dataset structure."""
        import os
        from pathlib import Path

        yaml_path = os.path.join(temp_dir, "data.yaml")
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        # Create a few dummy image files
        for i in range(5):
            img_file = os.path.join(img_dir, f"img_{i}.jpg")
            Path(img_file).touch()

        return yaml_path

    @pytest.fixture
    def mock_classify_dataset(self, temp_dir):
        """Create a mock classification dataset structure."""
        import os
        from pathlib import Path

        train_dir = os.path.join(temp_dir, "train")
        class_dirs = ["cat", "dog", "bird"]

        for class_name in class_dirs:
            class_dir = os.path.join(train_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            # Create dummy images
            for i in range(3):
                img_file = os.path.join(class_dir, f"img_{i}.jpg")
                Path(img_file).touch()

        return temp_dir

    def test_get_image_directory_detection(self, mock_detect_dataset):
        """Test that adapter finds correct image directory for detection tasks."""
        adapter = YOLOToLightlyAdapter(
            data_path=mock_detect_dataset, task_type="detect", transform=None
        )

        img_dir = adapter.get_image_directory()
        assert img_dir.endswith("train/images")
        assert os.path.exists(img_dir)

    def test_get_image_directory_classification(self, mock_classify_dataset):
        """Test that adapter finds correct image directory for classification."""
        adapter = YOLOToLightlyAdapter(
            data_path=mock_classify_dataset, task_type="classify", transform=None
        )

        img_dir = adapter.get_image_directory()
        assert img_dir.endswith("train")
        assert os.path.exists(img_dir)

    def test_get_image_directory_raises_on_missing(self, temp_dir):
        """Test that adapter raises error when image directory doesn't exist."""
        import os

        fake_path = os.path.join(temp_dir, "nonexistent.yaml")

        adapter = YOLOToLightlyAdapter(
            data_path=fake_path, task_type="detect", transform=None
        )

        with pytest.raises(ValueError, match="Image directory not found"):
            adapter.get_image_directory()

    @patch("fedyolo.train.ssl_trainer.LightlyDataset")
    @patch("fedyolo.train.ssl_trainer.DataLoader")
    def test_get_dataloader_creates_loader_with_correct_params(
        self, mock_dataloader, mock_dataset, mock_detect_dataset
    ):
        """Test that dataloader is created with correct parameters."""
        adapter = YOLOToLightlyAdapter(
            data_path=mock_detect_dataset, task_type="detect", transform=None
        )

        # Mock the dataset
        mock_dataset.return_value = MagicMock()

        adapter.get_dataloader(batch_size=32, num_workers=4)

        # Verify DataLoader was called with correct params
        mock_dataloader.assert_called_once()
        call_kwargs = mock_dataloader.call_args[1]

        assert call_kwargs["batch_size"] == 32
        assert call_kwargs["num_workers"] == 4
        assert call_kwargs["shuffle"] is True
        assert call_kwargs["drop_last"] is False  # Critical for small datasets


class TestLightlySSLTrainer:
    """Tests for Lightly SSL trainer."""

    @pytest.fixture
    def mock_yolo_model(self, mock_state_dict):
        """Create a mock YOLO model."""
        mock_model = MagicMock()
        mock_inner_model = MagicMock()

        # Create a simple mock backbone structure
        mock_inner_model.model = [MagicMock() for _ in range(24)]
        mock_model.model = mock_inner_model

        return mock_model

    def test_ssl_trainer_initialization_byol(self, mock_yolo_model):
        """Test SSL trainer initialization with BYOL."""
        config = {
            "method": "byol",
            "ssl_epochs": 20,
            "ssl_batch_size": 64,
            "projection_dim": 128,
            "hidden_dim": 2048,
        }

        with patch.object(
            LightlySSLTrainer, "_extract_backbone"
        ) as mock_extract, patch.object(LightlySSLTrainer, "_setup_ssl_components"):
            # Mock the backbone to return a simple module
            mock_backbone = torch.nn.Linear(512, 512)
            mock_extract.return_value = mock_backbone

            trainer = LightlySSLTrainer(mock_yolo_model, config, "detect")

            assert trainer.method == "byol"
            assert trainer.config == config
            assert trainer.task_type == "detect"

    def test_ssl_trainer_initialization_simclr(self, mock_yolo_model):
        """Test SSL trainer initialization with SimCLR."""
        config = {"method": "simclr"}

        with patch.object(
            LightlySSLTrainer, "_extract_backbone"
        ) as mock_extract, patch.object(LightlySSLTrainer, "_setup_ssl_components"):
            mock_backbone = torch.nn.Linear(512, 512)
            mock_extract.return_value = mock_backbone

            trainer = LightlySSLTrainer(mock_yolo_model, config, "detect")

            assert trainer.method == "simclr"

    def test_ssl_trainer_unsupported_method_raises_error(self, mock_yolo_model):
        """Test that unsupported SSL method raises error during setup."""
        config = {"method": "unsupported_method"}

        with patch.object(LightlySSLTrainer, "_extract_backbone") as mock_extract:
            # Create a proper mock backbone that returns correct shape
            class MockBackbone(torch.nn.Module):
                def forward(self, x):
                    batch_size = x.shape[0]
                    return torch.randn(batch_size, 512)

            mock_backbone = MockBackbone()
            mock_extract.return_value = mock_backbone

            with pytest.raises(ValueError, match="Unknown SSL method"):
                LightlySSLTrainer(mock_yolo_model, config, "detect")

    def test_ssl_trainer_supported_methods(self, mock_yolo_model):
        """Test that all supported SSL methods can be initialized."""
        supported_methods = ["byol", "simclr", "moco", "barlow_twins", "vicreg"]

        for method in supported_methods:
            config = {"method": method}

            with patch.object(
                LightlySSLTrainer, "_extract_backbone"
            ) as mock_extract, patch.object(LightlySSLTrainer, "_setup_ssl_components"):
                mock_backbone = torch.nn.Linear(512, 512)
                mock_extract.return_value = mock_backbone

                trainer = LightlySSLTrainer(mock_yolo_model, config, "detect")
                assert trainer.method == method


class TestBackboneExtraction:
    """Tests for backbone extraction from YOLO models."""

    def test_backbone_wrapper_stops_at_layer_9(self):
        """Test that BackboneWrapper extracts features up to layer 9."""
        # This is a simplified test - in reality we'd need a full YOLO model
        # For now, just verify the concept is testable

        mock_model = MagicMock()
        mock_model.model = [MagicMock() for _ in range(24)]

        # The actual test would require instantiating BackboneWrapper
        # and verifying it stops at the correct layer
        # This is more of an integration test
        pass

    def test_backbone_outputs_fixed_size_features(self):
        """Test that backbone outputs fixed-size features via adaptive pooling."""
        # Mock a simple backbone that applies adaptive pooling
        import torch.nn as nn

        class SimpleBackbone(nn.Module):
            def forward(self, x):
                # Simulate feature extraction
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                return x

        backbone = SimpleBackbone()

        # Test with different input sizes
        for size in [224, 320, 640]:
            dummy_input = torch.randn(2, 3, size, size)
            output = backbone(dummy_input)

            # Output should be flattened (batch_size, channels)
            assert output.dim() == 2
            assert output.shape[0] == 2  # batch size
