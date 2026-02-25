import os
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestDataLoaderConfiguration:
    """Tests for DataLoader configuration and drop_last settings."""

    @patch("fedyolo.train.ssl_trainer.LightlyDataset")
    @patch("fedyolo.train.ssl_trainer.DataLoader")
    def test_ssl_dataloader_drop_last_false(
        self, mock_dataloader, mock_dataset, temp_dir
    ):
        """Test that SSL DataLoader uses drop_last=False for small datasets."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Create mock dataset structure
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        # Create dummy images
        for i in range(5):
            Path(os.path.join(img_dir, f"img_{i}.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        # Create adapter and get dataloader
        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)
        mock_dataset.return_value = MagicMock()

        adapter.get_dataloader(batch_size=4, num_workers=0)

        # Verify drop_last=False was used
        call_kwargs = mock_dataloader.call_args[1]
        assert (
            call_kwargs["drop_last"] is False
        ), "SSL DataLoader should use drop_last=False to handle small datasets"

    @patch("fedyolo.train.ssl_trainer.DataLoader")
    def test_dataloader_batch_size_configuration(self, mock_dataloader, temp_dir):
        """Test that DataLoader respects batch_size parameter."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Setup mock dataset
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)
        Path(os.path.join(img_dir, "img_0.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)

        # Test different batch sizes
        for batch_size in [16, 32, 64]:
            with patch("fedyolo.train.ssl_trainer.LightlyDataset"):
                adapter.get_dataloader(batch_size=batch_size, num_workers=0)

                call_kwargs = mock_dataloader.call_args[1]
                assert call_kwargs["batch_size"] == batch_size

    @patch("fedyolo.train.ssl_trainer.DataLoader")
    @patch("fedyolo.train.ssl_trainer.LightlyDataset")
    def test_dataloader_shuffle_enabled(self, mock_dataset, mock_dataloader, temp_dir):
        """Test that DataLoader has shuffle enabled for training."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)
        Path(os.path.join(img_dir, "img_0.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)
        adapter.get_dataloader(batch_size=32, num_workers=0)

        call_kwargs = mock_dataloader.call_args[1]
        assert (
            call_kwargs["shuffle"] is True
        ), "DataLoader should have shuffle=True for SSL training"

    @patch("fedyolo.train.ssl_trainer.DataLoader")
    @patch("fedyolo.train.ssl_trainer.LightlyDataset")
    def test_dataloader_num_workers_configuration(
        self, mock_dataset, mock_dataloader, temp_dir
    ):
        """Test that DataLoader respects num_workers parameter."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)
        Path(os.path.join(img_dir, "img_0.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)

        # Test different worker counts
        for num_workers in [0, 2, 4]:
            adapter.get_dataloader(batch_size=32, num_workers=num_workers)

            call_kwargs = mock_dataloader.call_args[1]
            assert call_kwargs["num_workers"] == num_workers


class TestDatasetStructures:
    """Tests for different dataset structure handling."""

    def test_detection_dataset_structure(self, temp_dir):
        """Test that detection dataset structure is correctly identified."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Create detection dataset structure
        yaml_path = os.path.join(temp_dir, "data.yaml")
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)
        image_dir = adapter.get_image_directory()

        assert image_dir.endswith("train/images")
        assert os.path.exists(image_dir)

    def test_classification_dataset_structure_with_directory(self, temp_dir):
        """Test classification dataset with direct directory path."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Create classification dataset structure
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "class1"), exist_ok=True)

        adapter = YOLOToLightlyAdapter(temp_dir, "classify", transform=None)
        image_dir = adapter.get_image_directory()

        assert image_dir.endswith("train")
        assert os.path.exists(image_dir)

    def test_classification_dataset_structure_with_yaml(self, temp_dir):
        """Test classification dataset with yaml path."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Create structure
        yaml_path = os.path.join(temp_dir, "data.yaml")
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "class1"), exist_ok=True)

        adapter = YOLOToLightlyAdapter(yaml_path, "classify", transform=None)
        image_dir = adapter.get_image_directory()

        assert image_dir.endswith("train")
        assert os.path.exists(image_dir)

    def test_segmentation_dataset_structure(self, temp_dir):
        """Test that segmentation uses same structure as detection."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        yaml_path = os.path.join(temp_dir, "data.yaml")
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        adapter = YOLOToLightlyAdapter(yaml_path, "segment", transform=None)
        image_dir = adapter.get_image_directory()

        assert image_dir.endswith("train/images")

    def test_pose_dataset_structure(self, temp_dir):
        """Test that pose estimation uses same structure as detection."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        yaml_path = os.path.join(temp_dir, "data.yaml")
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        adapter = YOLOToLightlyAdapter(yaml_path, "pose", transform=None)
        image_dir = adapter.get_image_directory()

        assert image_dir.endswith("train/images")


class TestDataLoaderEdgeCases:
    """Tests for edge cases in data loading."""

    def test_empty_dataset_directory(self, temp_dir):
        """Test handling of empty dataset directory."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Create empty image directory
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        yaml_path = os.path.join(temp_dir, "data.yaml")

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)

        # Should find directory even if empty
        image_dir = adapter.get_image_directory()
        assert os.path.exists(image_dir)

    def test_small_dataset_batch_handling(self, temp_dir):
        """Test that small datasets with fewer images than batch size work correctly."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        # Create dataset with only 3 images
        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)

        for i in range(3):
            Path(os.path.join(img_dir, f"img_{i}.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)

        # With drop_last=False, this should handle batch_size > dataset_size
        with patch("fedyolo.train.ssl_trainer.LightlyDataset"), patch(
            "fedyolo.train.ssl_trainer.DataLoader"
        ) as mock_dataloader:
            adapter.get_dataloader(batch_size=10, num_workers=0)

            call_kwargs = mock_dataloader.call_args[1]
            assert (
                call_kwargs["drop_last"] is False
            ), "Small datasets require drop_last=False to avoid empty batches"


class TestTransformIntegration:
    """Tests for transform integration with DataLoader."""

    @patch("fedyolo.train.ssl_trainer.LightlyDataset")
    @patch("fedyolo.train.ssl_trainer.DataLoader")
    def test_dataloader_receives_transform(
        self, mock_dataloader, mock_dataset, temp_dir
    ):
        """Test that transforms are passed to the dataset correctly."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)
        Path(os.path.join(img_dir, "img_0.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        # Create a mock transform
        mock_transform = MagicMock()

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=mock_transform)
        adapter.get_dataloader(batch_size=32, num_workers=0)

        # Verify LightlyDataset was called with the transform
        mock_dataset.assert_called_once()
        call_kwargs = mock_dataset.call_args[1]
        assert call_kwargs["transform"] == mock_transform

    @patch("fedyolo.train.ssl_trainer.LightlyDataset")
    @patch("fedyolo.train.ssl_trainer.DataLoader")
    def test_dataloader_without_transform(
        self, mock_dataloader, mock_dataset, temp_dir
    ):
        """Test that DataLoader works without transforms (for testing)."""
        from fedyolo.train.ssl_trainer import YOLOToLightlyAdapter

        img_dir = os.path.join(temp_dir, "train", "images")
        os.makedirs(img_dir, exist_ok=True)
        Path(os.path.join(img_dir, "img_0.jpg")).touch()

        yaml_path = os.path.join(temp_dir, "data.yaml")

        adapter = YOLOToLightlyAdapter(yaml_path, "detect", transform=None)
        adapter.get_dataloader(batch_size=32, num_workers=0)

        # Should still create dataset, just without transform
        mock_dataset.assert_called_once()
        call_kwargs = mock_dataset.call_args[1]
        assert call_kwargs["transform"] is None
