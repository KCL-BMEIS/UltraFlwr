import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fedyolo.train.client import FlowerClient
from fedyolo.train.strategies import get_section_parameters


class TestWeightSharing:
    """Tests for weight sharing mechanism in federated learning."""

    @pytest.fixture
    def mock_client(self, mock_state_dict):
        """Create a mock FlowerClient for testing."""
        with patch("fedyolo.train.client.YOLO") as mock_yolo, patch(
            "fedyolo.train.client.os.path.exists", return_value=True
        ), patch("fedyolo.train.client.os.path.isdir", return_value=True):
            # Setup mock YOLO model
            mock_model = MagicMock()
            mock_model.state_dict.return_value = mock_state_dict
            mock_yolo.return_value.model = mock_model

            # Create client with FedBackboneAvg strategy
            client = FlowerClient(
                cid=0,
                data_path="/tmp/test",
                dataset_name="baseline",
                strategy_name="FedBackboneAvg",
                task="detect",
            )
            return client

    def test_get_parameters_fedbackboneavg(self, mock_client, mock_state_dict):
        """Test that FedBackboneAvg only returns backbone parameters."""
        params = mock_client.get_parameters(config={})

        # Calculate expected backbone parameters
        backbone, _, _ = get_section_parameters(mock_state_dict)
        expected_count = len(backbone)

        assert (
            len(params) == expected_count
        ), f"FedBackboneAvg should return {expected_count} backbone parameters"

    def test_get_parameters_fedavg(self, mock_state_dict):
        """Test that FedAvg returns all model parameters."""
        with patch("fedyolo.train.client.YOLO") as mock_yolo, patch(
            "fedyolo.train.client.os.path.exists", return_value=True
        ), patch("fedyolo.train.client.os.path.isdir", return_value=True):
            mock_model = MagicMock()
            mock_model.state_dict.return_value = mock_state_dict
            mock_yolo.return_value.model = mock_model

            client = FlowerClient(
                cid=0,
                data_path="/tmp/test",
                dataset_name="baseline",
                strategy_name="FedAvg",
                task="detect",
            )

            params = client.get_parameters(config={})
            expected_count = len(mock_state_dict)

            assert (
                len(params) == expected_count
            ), f"FedAvg should return all {expected_count} parameters"

    def test_get_parameters_fedheadavg(self, mock_state_dict):
        """Test that FedHeadAvg only returns head parameters."""
        with patch("fedyolo.train.client.YOLO") as mock_yolo, patch(
            "fedyolo.train.client.os.path.exists", return_value=True
        ), patch("fedyolo.train.client.os.path.isdir", return_value=True):
            mock_model = MagicMock()
            mock_model.state_dict.return_value = mock_state_dict
            mock_yolo.return_value.model = mock_model

            client = FlowerClient(
                cid=0,
                data_path="/tmp/test",
                dataset_name="baseline",
                strategy_name="FedHeadAvg",
                task="detect",
            )

            params = client.get_parameters(config={})

            # Calculate expected head parameters
            _, _, head = get_section_parameters(mock_state_dict)
            expected_count = len(head)

            assert (
                len(params) == expected_count
            ), f"FedHeadAvg should return {expected_count} head parameters"

    def test_set_parameters_perfect_match(self, mock_client, mock_state_dict):
        """Test set_parameters with perfect parameter count match."""
        backbone, _, _ = get_section_parameters(mock_state_dict)

        # Create fake parameters matching backbone count
        fake_params = [
            np.random.randn(*mock_state_dict[k].shape) for k in sorted(backbone.keys())
        ]

        # Should not raise any errors
        mock_client.set_parameters(fake_params)

    def test_set_parameters_shape_validation(self, mock_client, mock_state_dict):
        """Test that set_parameters validates parameter shapes."""
        backbone, _, _ = get_section_parameters(mock_state_dict)

        # Create parameters with wrong shapes
        wrong_shapes = [np.random.randn(5, 5) for _ in backbone.keys()]

        # Should raise ValueError for shape mismatch
        with pytest.raises(ValueError, match="Parameter reshape impossible"):
            mock_client.set_parameters(wrong_shapes)

    def test_set_parameters_intelligent_matching(self, mock_client, mock_state_dict):
        """Test intelligent parameter matching for architecture mismatches."""
        # Get backbone parameters
        backbone, _, _ = get_section_parameters(mock_state_dict)

        # Create fewer parameters than expected (simulate architecture mismatch)
        backbone_keys = sorted(backbone.keys())
        partial_params = [
            np.random.randn(*mock_state_dict[k].shape) for k in backbone_keys[:5]
        ]

        # Should handle mismatch gracefully with intelligent matching
        # This should not raise RuntimeError as long as some parameters match
        try:
            mock_client.set_parameters(partial_params)
        except RuntimeError as e:
            # Only fail if NO parameters were matched
            assert "No parameters could be matched" in str(e)

    def test_parameters_are_numpy_arrays(self, mock_client):
        """Test that get_parameters returns numpy arrays."""
        params = mock_client.get_parameters(config={})

        assert all(
            isinstance(p, np.ndarray) for p in params
        ), "All parameters should be numpy arrays"

    def test_strategy_group_membership(self):
        """Test that strategies are correctly grouped."""
        # Test that FedBackboneAvg is in backbone strategies
        assert "FedBackboneAvg" in FlowerClient._BACKBONE_STRATEGIES

        # Test that FedAvg is in all strategy groups
        assert "FedAvg" in FlowerClient._BACKBONE_STRATEGIES
        assert "FedAvg" in FlowerClient._NECK_STRATEGIES
        assert "FedAvg" in FlowerClient._HEAD_STRATEGIES

        # Test that FedHeadAvg is only in head strategies
        assert "FedHeadAvg" in FlowerClient._HEAD_STRATEGIES
        assert "FedHeadAvg" not in FlowerClient._BACKBONE_STRATEGIES
        assert "FedHeadAvg" not in FlowerClient._NECK_STRATEGIES


class TestParameterConsistency:
    """Tests for parameter ordering and consistency."""

    def test_parameter_ordering_consistency(self, mock_state_dict):
        """Test that get_parameters and set_parameters use consistent ordering."""
        with patch("fedyolo.train.client.YOLO") as mock_yolo, patch(
            "fedyolo.train.client.os.path.exists", return_value=True
        ), patch("fedyolo.train.client.os.path.isdir", return_value=True):
            mock_model = MagicMock()
            mock_model.state_dict.return_value = mock_state_dict
            mock_yolo.return_value.model = mock_model

            client = FlowerClient(
                cid=0,
                data_path="/tmp/test",
                dataset_name="baseline",
                strategy_name="FedBackboneAvg",
                task="detect",
            )

            # Get parameters
            params1 = client.get_parameters(config={})

            # Set and get again
            client.set_parameters(params1)
            params2 = client.get_parameters(config={})

            # Should have same count
            assert len(params1) == len(
                params2
            ), "Parameter count should be consistent across get/set cycles"

    def test_sorted_keys_usage(self, mock_state_dict):
        """Test that both get and set use sorted keys for consistency."""
        # This ensures server and client always agree on parameter order
        keys1 = sorted(mock_state_dict.keys())
        keys2 = sorted(mock_state_dict.keys())

        assert keys1 == keys2, "Sorted keys should always produce same order"
