"""
SSL (Self-Supervised Learning) Trainer Module

This module provides SSL training capabilities using the Lightly library.
Supports multiple SSL methods: BYOL, SimCLR, MoCo, Barlow Twins, VICReg.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Optional

# Lightly imports
from lightly.data import LightlyDataset  # type: ignore
from lightly.transforms import SimCLRTransform, BYOLTransform, MoCoV2Transform  # type: ignore
from lightly.models.modules import (  # type: ignore
    SimCLRProjectionHead,
    BYOLProjectionHead,
    BYOLPredictionHead,
    MoCoProjectionHead,
)
from lightly.loss import (  # type: ignore
    NTXentLoss,
    NegativeCosineSimilarity,
    BarlowTwinsLoss,
    VICRegLoss,
)

logger = logging.getLogger(__name__)


class YOLOToLightlyAdapter:
    """Adapts YOLO dataset format to Lightly-compatible DataLoader."""

    def __init__(self, data_path, task_type, transform=None):
        """
        Initialize the adapter.

        Args:
            data_path: Path to the dataset (YOLO format or classification format)
            task_type: Type of task (detect, segment, pose, classify)
            transform: Lightly transform for augmentations
        """
        self.data_path = data_path
        self.task_type = task_type
        self.transform = transform

    def get_image_directory(self):
        """Get the directory containing images based on task type."""
        if self.task_type == "classify":
            # Classification: data_path can be either a directory or data.yaml
            if self.data_path.endswith(".yaml") or self.data_path.endswith(".yml"):
                # data.yaml path - extract train directory from parent
                yaml_parent = Path(self.data_path).parent
                img_dir = os.path.join(yaml_parent, "train")
            else:
                # Direct directory path (legacy)
                img_dir = os.path.join(self.data_path, "train")
        else:
            # Detection/Segmentation/Pose: data_path points to data.yaml
            # Images are in train/images/ directory relative to data.yaml parent
            yaml_parent = Path(self.data_path).parent
            img_dir = os.path.join(yaml_parent, "train", "images")

        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")

        return img_dir

    def get_dataloader(self, batch_size=64, num_workers=0):
        """
        Create a DataLoader for SSL training.

        Args:
            batch_size: Batch size for training
            num_workers: Number of workers for data loading

        Returns:
            DataLoader for SSL training
        """
        img_dir = self.get_image_directory()

        # Create Lightly dataset
        dataset = LightlyDataset(input_dir=img_dir, transform=self.transform)  # type: ignore

        # Create DataLoader
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,  # Allow smaller final batch for small datasets
        )

        return dataloader


class LightlySSLTrainer:
    """Handles SSL training using Lightly library."""

    def __init__(self, model, config, task_type):
        """
        Initialize SSL trainer.

        Args:
            model: YOLO model (we'll extract the backbone)
            config: SSL configuration dictionary
            task_type: Type of task (detect, segment, pose, classify)
        """
        self.model = model
        self.config = config
        self.task_type = task_type
        self.method = config.get("method", "byol")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract backbone from YOLO model
        # YOLO model structure: model.model (the actual network)
        # We want just the backbone (feature extractor)
        self.backbone = self._extract_backbone()

        # Setup SSL components based on method
        self.projection_head: Optional[nn.Module] = None
        self.prediction_head: Optional[nn.Module] = None  # Only for BYOL
        self.criterion: Optional[nn.Module] = None
        self.transform: Optional[object] = None

        self._setup_ssl_components()

    def _extract_backbone(self):
        """Extract the backbone (feature extractor) from YOLO model."""
        try:
            # YOLO model structure varies by task, but generally:
            # model.model contains the actual network
            # The backbone is typically the first part (before detection/seg heads)

            # For now, we'll use the entire model and add an adapter
            # to extract features before the final layers
            yolo_model = self.model.model

            # Create a feature extractor that stops before task-specific heads
            # This is a simplified approach - we'll use the model up to a certain point
            class BackboneWrapper(nn.Module):
                def __init__(self, yolo_model):
                    super().__init__()
                    self.model = yolo_model

                def forward(self, x):
                    # Forward through YOLO model and extract features
                    # This gets features before the final detection/classification layers
                    # We'll use the backbone layers (typically indices 0-9 in YOLO)
                    for i, layer in enumerate(self.model.model):
                        x = layer(x)
                        # Stop before the head layers (usually after backbone)
                        # YOLO11 typically has backbone in layers 0-9
                        if i == 9:  # Adjust based on architecture
                            break
                    # Global average pooling to get fixed-size features
                    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                    x = torch.flatten(x, 1)
                    return x

            return BackboneWrapper(yolo_model)

        except Exception as e:
            logger.error(f"Failed to extract backbone: {e}")
            raise

    def _setup_ssl_components(self):
        """Setup SSL components based on the selected method."""
        # Move backbone to device first
        self.backbone = self.backbone.to(self.device)

        # Feature dimension - this depends on YOLO backbone
        # For YOLO11n, typical feature dimension is 512
        # We'll determine this dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        logger.info(f"Detected backbone feature dimension: {feature_dim}")

        projection_dim = self.config.get("projection_dim", 128)
        hidden_dim = self.config.get("hidden_dim", 2048)

        if self.method == "simclr":
            self.projection_head = SimCLRProjectionHead(
                input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim
            )
            self.criterion = NTXentLoss(temperature=self.config.get("temperature", 0.5))
            self.transform = SimCLRTransform(input_size=224)

        elif self.method == "byol":
            self.projection_head = BYOLProjectionHead(
                input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim
            )
            self.prediction_head = BYOLPredictionHead(
                input_dim=projection_dim,
                hidden_dim=hidden_dim,
                output_dim=projection_dim,
            )
            self.criterion = NegativeCosineSimilarity()
            # Use default BYOL transforms (creates two views automatically)
            self.transform = BYOLTransform()

        elif self.method == "moco":
            self.projection_head = MoCoProjectionHead(
                input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim
            )
            self.criterion = NTXentLoss(temperature=self.config.get("temperature", 0.5))
            self.transform = MoCoV2Transform(input_size=224)

        elif self.method == "barlow_twins":
            self.projection_head = SimCLRProjectionHead(  # Same as SimCLR
                input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim
            )
            self.criterion = BarlowTwinsLoss()
            self.transform = SimCLRTransform(input_size=224)

        elif self.method == "vicreg":
            self.projection_head = SimCLRProjectionHead(
                input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim
            )
            self.criterion = VICRegLoss()
            self.transform = SimCLRTransform(input_size=224)

        else:
            raise ValueError(f"Unknown SSL method: {self.method}")

        # Move projection/prediction heads to device (backbone already moved earlier)
        self.projection_head = self.projection_head.to(self.device)
        if self.prediction_head is not None:
            self.prediction_head = self.prediction_head.to(self.device)

        logger.info(f"SSL method '{self.method}' initialized successfully")

    def pretrain(self, data_path, client_id, epochs=None):
        """
        Run SSL pretraining.

        Args:
            data_path: Path to the dataset
            client_id: Client ID for logging
            epochs: Number of epochs (overrides config if provided)
        """
        epochs = epochs or self.config.get("ssl_epochs", 20)
        batch_size = self.config.get("ssl_batch_size", 64)

        logger.info(f"Client {client_id}: Starting SSL pretraining with {self.method}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

        # Create dataset adapter
        adapter = YOLOToLightlyAdapter(data_path, self.task_type, self.transform)
        dataloader = adapter.get_dataloader(batch_size=batch_size, num_workers=0)

        # Debug: Check dataloader size
        logger.info(
            f"Client {client_id}: Dataloader created with {len(dataloader)} batches"  # type: ignore
        )
        logger.info(
            f"Client {client_id}: Dataset size: {len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 'unknown'}"  # type: ignore
        )

        # Setup optimizer
        params = list(self.backbone.parameters()) + list(
            self.projection_head.parameters()  # type: ignore
        )
        if self.prediction_head is not None:
            params += list(self.prediction_head.parameters())

        optimizer = torch.optim.Adam(params, lr=1e-4)

        # Training loop
        self.backbone.train()
        self.projection_head.train()  # type: ignore
        if self.prediction_head is not None:
            self.prediction_head.train()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx == 0:  # Log first batch
                    logger.info(
                        f"Client {client_id} Epoch {epoch + 1}: Processing batch {batch_idx}"
                    )

                # Lightly returns (views, labels, filenames)
                # views is a list of augmented views
                views = batch[0]

                # Get two views
                x0, x1 = views[0].to(self.device), views[1].to(self.device)

                # Forward pass
                f0 = self.backbone(x0)
                f1 = self.backbone(x1)

                z0 = self.projection_head(f0)  # type: ignore
                z1 = self.projection_head(f1)  # type: ignore

                # Apply prediction head if using BYOL
                if self.prediction_head is not None:
                    p0 = self.prediction_head(z0)
                    p1 = self.prediction_head(z1)
                    loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))  # type: ignore
                else:
                    loss = self.criterion(z0, z1)  # type: ignore

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            logger.info(
                f"Client {client_id} SSL Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}"
            )

        logger.info(f"Client {client_id}: SSL pretraining complete")

        # Save client-specific SSL checkpoint after training
        self._save_client_checkpoint(client_id)

    def _save_client_checkpoint(self, client_id):
        """Save client-specific SSL checkpoint."""
        from fedyolo.config import get_output_dirs

        output_dirs = get_output_dirs()
        client_ssl_dir = os.path.join(
            output_dirs["checkpoints_ssl_clients"], f"client_{client_id}"
        )
        os.makedirs(client_ssl_dir, exist_ok=True)

        checkpoint_path = os.path.join(client_ssl_dir, "ssl_backbone.pt")

        # Save only the backbone (not projection/prediction heads)
        checkpoint = {
            "backbone_state_dict": self.backbone.state_dict(),
            "method": self.method,
            "task_type": self.task_type,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Client {client_id}: Saved SSL checkpoint to {checkpoint_path}")
