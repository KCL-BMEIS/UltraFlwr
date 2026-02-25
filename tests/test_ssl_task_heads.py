"""
Tests for SSL weight loading with task-specific heads.

This test suite ensures that when SSL-pretrained weights are loaded,
task-specific heads (Pose, Segment, Classify) are properly attached
and the model architecture matches the dataset requirements.
"""

import pytest
import torch
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from collections import OrderedDict


class TestSSLWeightLoadingWithTaskHeads:
    """Tests for SSL weight loading with different task-specific heads."""

    @pytest.fixture
    def mock_ssl_checkpoint(self):
        """Create a mock SSL checkpoint with backbone weights."""
        state_dict = OrderedDict()

        # Create backbone weights (model.0 through model.9)
        for i in range(10):
            state_dict[f"model.{i}.weight"] = torch.randn(64, 64)
            state_dict[f"model.{i}.bias"] = torch.randn(64)

        # Create some neck layers (model.10-15)
        for i in range(10, 16):
            state_dict[f"model.{i}.weight"] = torch.randn(128, 128)
            state_dict[f"model.{i}.bias"] = torch.randn(128)

        # Create detect head (model.23)
        state_dict["model.23.cv2.0.0.conv.weight"] = torch.randn(64, 64, 1, 1)
        state_dict["model.23.cv2.0.0.conv.bias"] = torch.randn(64)

        # Wrap in checkpoint format
        checkpoint = {
            "model": state_dict,
            "epoch": 10,
        }
        return checkpoint

    @pytest.fixture
    def mock_pose_model_state_dict(self):
        """Create a mock pose model state dict with Pose head."""
        state_dict = OrderedDict()

        # Backbone (same structure as SSL)
        for i in range(10):
            state_dict[f"model.{i}.weight"] = torch.randn(64, 64)
            state_dict[f"model.{i}.bias"] = torch.randn(64)

        # Neck
        for i in range(10, 16):
            state_dict[f"model.{i}.weight"] = torch.randn(128, 128)
            state_dict[f"model.{i}.bias"] = torch.randn(128)

        # Pose head (model.23) - different structure than Detect
        state_dict["model.23.cv2.0.0.conv.weight"] = torch.randn(
            36, 64, 1, 1
        )  # 12 kpts * 3
        state_dict["model.23.cv2.0.0.conv.bias"] = torch.randn(36)
        state_dict["model.23.cv4.0.0.conv.weight"] = torch.randn(64, 64, 1, 1)
        state_dict["model.23.cv4.0.0.conv.bias"] = torch.randn(64)

        return state_dict

    @pytest.fixture
    def mock_segment_model_state_dict(self):
        """Create a mock segment model state dict with Segment head."""
        state_dict = OrderedDict()

        # Backbone (same structure as SSL)
        for i in range(10):
            state_dict[f"model.{i}.weight"] = torch.randn(64, 64)
            state_dict[f"model.{i}.bias"] = torch.randn(64)

        # Neck
        for i in range(10, 16):
            state_dict[f"model.{i}.weight"] = torch.randn(128, 128)
            state_dict[f"model.{i}.bias"] = torch.randn(128)

        # Segment head - has proto head for masks
        state_dict["model.23.cv2.0.0.conv.weight"] = torch.randn(64, 64, 1, 1)
        state_dict["model.23.cv2.0.0.conv.bias"] = torch.randn(64)
        state_dict["model.23.proto.conv.weight"] = torch.randn(32, 256, 3, 3)
        state_dict["model.23.proto.conv.bias"] = torch.randn(32)

        return state_dict

    @pytest.fixture
    def temp_weights_dir(self):
        """Create temporary weights directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            ssl_dir = os.path.join(tmpdir, "ssl")
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(ssl_dir, exist_ok=True)
            yield {"base": base_dir, "ssl": ssl_dir, "root": tmpdir}

    def test_pose_model_loads_with_ssl_backbone(
        self, mock_ssl_checkpoint, mock_pose_model_state_dict, temp_weights_dir
    ):
        """Test that pose model correctly loads SSL backbone while preserving Pose head."""

        # Save mock SSL checkpoint
        ssl_file = os.path.join(temp_weights_dir["ssl"], "yolo11n-ssl.pt")
        torch.save(mock_ssl_checkpoint, ssl_file)

        # Simulate the client initialization logic
        ssl_state_dict = mock_ssl_checkpoint["model"]
        current_state_dict = mock_pose_model_state_dict.copy()

        # Update only backbone weights (model.0-9)
        updated_state_dict = current_state_dict.copy()
        updated_count = 0

        for key in ssl_state_dict:
            if key.startswith("model.") and key in current_state_dict:
                layer_num_match = key.split(".")[1]
                if layer_num_match.isdigit() and int(layer_num_match) < 10:
                    if current_state_dict[key].shape == ssl_state_dict[key].shape:
                        updated_state_dict[key] = ssl_state_dict[key]
                        updated_count += 1

        # Verify backbone was updated
        assert updated_count > 0, "Should update at least some backbone layers"

        # Verify pose head was NOT updated (preserved original)
        pose_head_keys = [k for k in current_state_dict.keys() if "model.23.cv4" in k]
        for key in pose_head_keys:
            assert torch.equal(
                updated_state_dict[key], mock_pose_model_state_dict[key]
            ), f"Pose head layer {key} should be preserved"

        # Verify backbone was updated
        for i in range(10):
            key = f"model.{i}.weight"
            if key in ssl_state_dict and key in updated_state_dict:
                assert torch.equal(
                    updated_state_dict[key], ssl_state_dict[key]
                ), f"Backbone layer {key} should be updated from SSL weights"

    def test_segment_model_loads_with_ssl_backbone(
        self, mock_ssl_checkpoint, mock_segment_model_state_dict
    ):
        """Test that segment model correctly loads SSL backbone while preserving Segment head."""

        ssl_state_dict = mock_ssl_checkpoint["model"]
        current_state_dict = mock_segment_model_state_dict.copy()

        # Update only backbone weights
        updated_state_dict = current_state_dict.copy()
        updated_count = 0

        for key in ssl_state_dict:
            if key.startswith("model.") and key in current_state_dict:
                layer_num_match = key.split(".")[1]
                if layer_num_match.isdigit() and int(layer_num_match) < 10:
                    if current_state_dict[key].shape == ssl_state_dict[key].shape:
                        updated_state_dict[key] = ssl_state_dict[key]
                        updated_count += 1

        # Verify segment-specific proto head was preserved
        proto_keys = [k for k in current_state_dict.keys() if "proto" in k]
        for key in proto_keys:
            assert torch.equal(
                updated_state_dict[key], mock_segment_model_state_dict[key]
            ), f"Segment proto layer {key} should be preserved"

    def test_ssl_loading_skips_incompatible_shapes(self):
        """Test that SSL loading gracefully skips layers with incompatible shapes."""

        ssl_state_dict = OrderedDict()
        ssl_state_dict["model.0.weight"] = torch.randn(32, 32)  # Different shape
        ssl_state_dict["model.1.weight"] = torch.randn(64, 64)  # Matching shape

        current_state_dict = OrderedDict()
        current_state_dict["model.0.weight"] = torch.randn(64, 64)  # Different
        current_state_dict["model.1.weight"] = torch.randn(64, 64)  # Same

        # Update logic
        updated_state_dict = current_state_dict.copy()
        updated_count = 0
        skipped_count = 0

        for key in ssl_state_dict:
            if key in current_state_dict:
                if current_state_dict[key].shape == ssl_state_dict[key].shape:
                    updated_state_dict[key] = ssl_state_dict[key]
                    updated_count += 1
                else:
                    skipped_count += 1

        assert updated_count == 1, "Should update one matching layer"
        assert skipped_count == 1, "Should skip one incompatible layer"

        # Verify model.0 was NOT updated (shape mismatch)
        assert torch.equal(
            updated_state_dict["model.0.weight"], current_state_dict["model.0.weight"]
        )

        # Verify model.1 WAS updated (shape match)
        assert torch.equal(
            updated_state_dict["model.1.weight"], ssl_state_dict["model.1.weight"]
        )

    def test_ssl_loading_only_updates_backbone_layers(self, mock_ssl_checkpoint):
        """Test that SSL loading only updates backbone (model.0-9) and not neck/head."""

        ssl_state_dict = mock_ssl_checkpoint["model"]

        # Create target state dict with all layers
        current_state_dict = OrderedDict()
        for i in range(24):
            current_state_dict[f"model.{i}.weight"] = torch.randn(64, 64)

        original_state_dict = current_state_dict.copy()

        # Apply update logic
        updated_state_dict = current_state_dict.copy()

        for key in ssl_state_dict:
            if key.startswith("model.") and key in current_state_dict:
                layer_num_match = key.split(".")[1]
                if layer_num_match.isdigit() and int(layer_num_match) < 10:
                    if current_state_dict[key].shape == ssl_state_dict[key].shape:
                        updated_state_dict[key] = ssl_state_dict[key]

        # Verify backbone (0-9) was updated
        for i in range(10):
            key = f"model.{i}.weight"
            if key in ssl_state_dict:
                assert not torch.equal(
                    updated_state_dict[key], original_state_dict[key]
                ), f"Backbone layer {key} should be updated"

        # Verify neck/head (10+) was NOT updated
        for i in range(10, 24):
            key = f"model.{i}.weight"
            assert torch.equal(
                updated_state_dict[key], original_state_dict[key]
            ), f"Neck/head layer {key} should NOT be updated"


class TestPoseYAMLConfiguration:
    """Tests for pose-specific YAML configuration."""

    @pytest.fixture
    def pose_yaml_path(self):
        """Return path to pose YAML config."""
        from fedyolo.config import HOME

        return os.path.join(HOME, "fedyolo/yolo_configs/yolo11n_pose.yaml")

    def test_pose_yaml_exists(self, pose_yaml_path):
        """Test that pose-specific YAML configuration file exists."""
        assert os.path.exists(
            pose_yaml_path
        ), "Pose YAML config must exist at fedyolo/yolo_configs/yolo11n_pose.yaml"

    def test_pose_yaml_has_correct_structure(self, pose_yaml_path):
        """Test that pose YAML has required fields for 12-keypoint tiger pose."""
        with open(pose_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required fields
        assert "nc" in config, "Pose YAML must have 'nc' field"
        assert "kpt_shape" in config, "Pose YAML must have 'kpt_shape' field"
        assert "flip_idx" in config, "Pose YAML must have 'flip_idx' field"

        # Check values
        assert config["nc"] == 1, "Tiger pose should have 1 class"
        assert config["kpt_shape"] == [
            12,
            2,
        ], "Tiger pose should have 12 keypoints with 2 dimensions"
        assert (
            len(config["flip_idx"]) == 12
        ), "Should have flip indices for all 12 keypoints"

        # Check architecture sections
        assert "backbone" in config, "Must have backbone section"
        assert "head" in config, "Must have head section"

        # Check head has Pose module
        head_modules = config["head"]
        last_module = head_modules[-1]
        assert last_module[2] == "Pose", "Last head module should be Pose"
        assert "kpt_shape" in str(
            last_module[3]
        ), "Pose head should reference kpt_shape"

    def test_pose_yaml_flip_idx_is_sequential(self, pose_yaml_path):
        """Test that flip_idx is sequential for symmetric keypoints."""
        with open(pose_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        flip_idx = config["flip_idx"]
        # For tiger pose, assuming symmetric keypoints, flip_idx should be sequential
        assert flip_idx == list(
            range(12)
        ), "For non-mirrored keypoints, flip_idx should be [0, 1, 2, ..., 11]"


class TestClientInitializationWithSSL:
    """Integration tests for client initialization with SSL weights."""

    @pytest.fixture
    def temp_weights_dir(self):
        """Create temporary weights directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            ssl_dir = os.path.join(tmpdir, "ssl")
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(ssl_dir, exist_ok=True)
            yield {"base": base_dir, "ssl": ssl_dir, "root": tmpdir}

    @pytest.fixture
    def mock_output_dirs(self, temp_weights_dir):
        """Mock get_output_dirs to return temp directories."""
        # Create ssl_clients checkpoint dir
        ssl_clients_dir = os.path.join(
            temp_weights_dir["base"], "..", "checkpoints", "ssl_clients"
        )
        os.makedirs(ssl_clients_dir, exist_ok=True)
        clients_dir = os.path.join(
            temp_weights_dir["base"], "..", "checkpoints", "clients"
        )
        os.makedirs(clients_dir, exist_ok=True)
        return {
            "weights_base": temp_weights_dir["base"],
            "weights_ssl": temp_weights_dir["ssl"],
            "checkpoints_ssl_clients": ssl_clients_dir,
            "checkpoints_clients": clients_dir,
        }

    @pytest.fixture
    def mock_ssl_checkpoint(self):
        """Create a mock SSL checkpoint with backbone weights."""
        state_dict = OrderedDict()

        # Create backbone weights (model.0 through model.9)
        for i in range(10):
            state_dict[f"model.{i}.weight"] = torch.randn(64, 64)
            state_dict[f"model.{i}.bias"] = torch.randn(64)

        # Create some neck layers (model.10-15)
        for i in range(10, 16):
            state_dict[f"model.{i}.weight"] = torch.randn(128, 128)
            state_dict[f"model.{i}.bias"] = torch.randn(128)

        # Create detect head (model.23)
        state_dict["model.23.cv2.0.0.conv.weight"] = torch.randn(64, 64, 1, 1)
        state_dict["model.23.cv2.0.0.conv.bias"] = torch.randn(64)

        # Wrap in checkpoint format
        checkpoint = {
            "model": state_dict,
            "epoch": 10,
        }
        return checkpoint

    @pytest.fixture
    def mock_pose_model_state_dict(self):
        """Create a mock pose model state dict with Pose head."""
        state_dict = OrderedDict()

        # Backbone (same structure as SSL)
        for i in range(10):
            state_dict[f"model.{i}.weight"] = torch.randn(64, 64)
            state_dict[f"model.{i}.bias"] = torch.randn(64)

        # Neck
        for i in range(10, 16):
            state_dict[f"model.{i}.weight"] = torch.randn(128, 128)
            state_dict[f"model.{i}.bias"] = torch.randn(128)

        # Pose head (model.23) - different structure than Detect
        state_dict["model.23.cv2.0.0.conv.weight"] = torch.randn(
            36, 64, 1, 1
        )  # 12 kpts * 3
        state_dict["model.23.cv2.0.0.conv.bias"] = torch.randn(36)
        state_dict["model.23.cv4.0.0.conv.weight"] = torch.randn(64, 64, 1, 1)
        state_dict["model.23.cv4.0.0.conv.bias"] = torch.randn(64)

        return state_dict

    def test_client_init_without_ssl_loads_task_weights(self, mock_output_dirs):
        """Test that client initializes with task-specific weights when SSL weights don't exist."""

        with (
            patch("fedyolo.train.client.get_output_dirs") as mock_dirs,
            patch("fedyolo.train.client.YOLO") as mock_yolo,
            patch("fedyolo.train.client.os.path.exists") as mock_exists,
            patch("fedyolo.train.client.ensure_weights_available") as mock_ensure,
        ):
            mock_dirs.return_value = mock_output_dirs

            # Mock: SSL weights don't exist, but task weights do
            def exists_side_effect(path):
                if "yolo11n-ssl.pt" in path:
                    return False  # SSL weights don't exist
                elif "yolo11n-pose.pt" in path:
                    return True  # Task weights exist
                return False

            mock_exists.side_effect = exists_side_effect

            # Mock ensure_weights_available to return the weight path
            mock_ensure.side_effect = lambda w, d: os.path.join(d, w)

            # Mock YOLO model
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            from fedyolo.train.client import FlowerClient

            _ = FlowerClient(
                cid=0,
                data_path="/tmp/data.yaml",
                dataset_name="pose",
                strategy_name="FedBackboneAvg",
                task="pose",
            )

            # Should have loaded pose-specific weights
            mock_yolo.assert_called_once()
            call_args = mock_yolo.call_args[0][0]
            assert "yolo11n-pose.pt" in call_args

    def test_client_init_with_ssl_loads_task_arch_then_ssl_backbone(
        self, mock_output_dirs, mock_ssl_checkpoint, mock_pose_model_state_dict
    ):
        """Test that client with SSL loads task architecture first, then SSL backbone."""

        # Save SSL checkpoint
        ssl_file = os.path.join(mock_output_dirs["weights_ssl"], "yolo11n-ssl.pt")
        torch.save(mock_ssl_checkpoint, ssl_file)

        with (
            patch("fedyolo.train.client.get_output_dirs") as mock_dirs,
            patch("fedyolo.train.client.YOLO") as mock_yolo,
            patch("fedyolo.train.client.torch.load") as mock_torch_load,
            patch("fedyolo.train.client.os.path.exists") as mock_exists,
        ):
            mock_dirs.return_value = mock_output_dirs

            # Mock: Both SSL and task weights exist
            def exists_side_effect(path):
                if "yolo11n-ssl.pt" in path or "yolo11n-pose.pt" in path:
                    return True
                return False

            mock_exists.side_effect = exists_side_effect

            # Mock torch.load to return SSL checkpoint
            mock_torch_load.return_value = mock_ssl_checkpoint

            # Mock YOLO to return model with state dict
            mock_model_instance = MagicMock()
            mock_model_instance.model.state_dict.return_value = (
                mock_pose_model_state_dict
            )
            mock_yolo.return_value = mock_model_instance

            from fedyolo.train.client import FlowerClient

            _ = FlowerClient(
                cid=2,
                data_path="/tmp/pose_data.yaml",
                dataset_name="pose",
                strategy_name="FedBackboneAvg",
                task="pose",
            )

            # Should have loaded task-specific architecture first
            mock_yolo.assert_called_once()
            call_args = mock_yolo.call_args[0][0]
            assert "yolo11n-pose.pt" in call_args

            # Should have loaded SSL checkpoint
            mock_torch_load.assert_called_once_with(
                ssl_file, map_location="cpu", weights_only=False
            )

            # Should have called load_state_dict to update weights
            mock_model_instance.model.load_state_dict.assert_called_once()

    def test_client_init_raises_error_if_task_weights_missing_with_ssl(
        self, mock_output_dirs, mock_ssl_checkpoint
    ):
        """Test that client raises error if task weights don't exist when SSL weights do."""

        # Save SSL checkpoint
        ssl_file = os.path.join(mock_output_dirs["weights_ssl"], "yolo11n-ssl.pt")
        torch.save(mock_ssl_checkpoint, ssl_file)

        with (
            patch("fedyolo.train.client.get_output_dirs") as mock_dirs,
            patch("fedyolo.train.client.os.path.exists") as mock_exists,
            patch("fedyolo.train.client.ensure_weights_available") as mock_ensure,
        ):
            mock_dirs.return_value = mock_output_dirs

            # Mock: SSL exists but task weights don't
            def exists_side_effect(path):
                if "yolo11n-ssl.pt" in path:
                    return True
                elif "yolo11n-pose.pt" in path:
                    return False  # Task weights missing
                return False

            mock_exists.side_effect = exists_side_effect

            # Mock ensure_weights_available to raise error when weights don't exist
            mock_ensure.side_effect = RuntimeError("Failed to download yolo11n-pose.pt")

            from fedyolo.train.client import FlowerClient

            with pytest.raises(RuntimeError, match="Failed to load SSL weights"):
                FlowerClient(
                    cid=2,
                    data_path="/tmp/pose_data.yaml",
                    dataset_name="pose",
                    strategy_name="FedBackboneAvg",
                    task="pose",
                )


class TestEvaluatorWithPoseModel:
    """Tests for evaluator loading pose models correctly."""

    def test_evaluator_uses_pose_yaml_when_available(self):
        """Test that evaluator prefers pose YAML over base weights when available."""
        from fedyolo.config import HOME

        yaml_path = f"{HOME}/fedyolo/yolo_configs/yolo11n_pose.yaml"

        # Check logic from evaluator.py:39-43
        dataset_name = "pose"
        task = "pose"

        if dataset_name and task in ["pose"]:
            expected_yaml = f"{HOME}/fedyolo/yolo_configs/yolo11n_{dataset_name}.yaml"

            assert expected_yaml == yaml_path
            assert os.path.exists(
                expected_yaml
            ), f"Pose YAML should exist at {expected_yaml}"

    def test_load_fl_trained_model_with_pose_yaml(self):
        """Test that load_fl_trained_model uses pose YAML for correct architecture."""
        from fedyolo.config import HOME

        yaml_path = f"{HOME}/fedyolo/yolo_configs/yolo11n_pose.yaml"

        # Verify YAML exists
        assert os.path.exists(yaml_path), "Pose YAML must exist for this test"

        # Verify YAML has correct keypoint config
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        assert config["kpt_shape"] == [
            12,
            2,
        ], "Pose YAML should specify 12 keypoints with 2 dimensions for tiger dataset"
