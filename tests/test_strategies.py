from typing import Any
from fedyolo.train import strategies as strategies_module  # type: ignore
from fedyolo.train.strategies import (
    get_section_parameters,
    create_partial_strategy,
)
import flwr as fl

# Dynamically load strategy classes to avoid type: ignore on imports
_strategies_dict: dict[str, Any] = {}
for name in [
    "FedAvg",
    "FedBackboneAvg",
    "FedNeckAvg",
    "FedHeadAvg",
    "FedBackboneNeckAvg",
]:
    _strategies_dict[name] = getattr(strategies_module, name, None)

FedAvg = _strategies_dict["FedAvg"]  # type: ignore
FedBackboneAvg = _strategies_dict["FedBackboneAvg"]  # type: ignore
FedNeckAvg = _strategies_dict["FedNeckAvg"]  # type: ignore
FedHeadAvg = _strategies_dict["FedHeadAvg"]  # type: ignore
FedBackboneNeckAvg = _strategies_dict["FedBackboneNeckAvg"]  # type: ignore


class TestGetSectionParameters:
    """Tests for get_section_parameters function."""

    def test_get_section_parameters_basic(self, mock_state_dict):
        """Test that state dict is correctly split into backbone, neck, and head."""
        backbone, neck, head = get_section_parameters(mock_state_dict)

        # Check that all sections are non-empty
        assert len(backbone) > 0
        assert len(neck) > 0
        assert len(head) > 0

        # Check that total params match original
        total_params = len(backbone) + len(neck) + len(head)
        assert total_params == len(mock_state_dict)

    def test_backbone_contains_correct_layers(self, mock_state_dict):
        """Test that backbone contains layers 0-8."""
        backbone, _, _ = get_section_parameters(mock_state_dict)

        # All backbone keys should start with model.0 through model.8
        for key in backbone.keys():
            layer_idx = int(key.split(".")[1])
            assert (
                layer_idx < 9
            ), f"Backbone should only contain layers 0-8, found {key}"

    def test_neck_contains_correct_layers(self, mock_state_dict):
        """Test that neck contains layers 9-22."""
        _, neck, _ = get_section_parameters(mock_state_dict)

        # All neck keys should start with model.9 through model.22
        for key in neck.keys():
            layer_idx = int(key.split(".")[1])
            assert (
                9 <= layer_idx < 23
            ), f"Neck should only contain layers 9-22, found {key}"

    def test_head_contains_correct_layers(self, mock_state_dict):
        """Test that head contains layer 23."""
        _, _, head = get_section_parameters(mock_state_dict)

        # All head keys should start with model.23
        for key in head.keys():
            assert key.startswith(
                "model.23"
            ), f"Head should only contain layer 23, found {key}"

    def test_no_parameter_overlap(self, mock_state_dict):
        """Test that there's no overlap between sections."""
        backbone, neck, head = get_section_parameters(mock_state_dict)

        backbone_keys = set(backbone.keys())
        neck_keys = set(neck.keys())
        head_keys = set(head.keys())

        # Check no overlap
        assert (
            len(backbone_keys & neck_keys) == 0
        ), "Backbone and neck should not overlap"
        assert (
            len(backbone_keys & head_keys) == 0
        ), "Backbone and head should not overlap"
        assert len(neck_keys & head_keys) == 0, "Neck and head should not overlap"


class TestCreatePartialStrategy:
    """Tests for create_partial_strategy factory function."""

    def test_create_full_strategy(self):
        """Test creating a strategy that updates all sections."""
        FullStrategy = create_partial_strategy(
            fl.server.strategy.FedAvg,
            "TestFedAvg",
            update_backbone=True,
            update_neck=True,
            update_head=True,
            docstring="Test full strategy",
        )

        assert FullStrategy.update_backbone is True
        assert FullStrategy.update_neck is True
        assert FullStrategy.update_head is True

    def test_create_backbone_only_strategy(self):
        """Test creating a strategy that only updates backbone."""
        BackboneStrategy = create_partial_strategy(
            fl.server.strategy.FedAvg,
            "TestFedBackboneAvg",
            update_backbone=True,
            update_neck=False,
            update_head=False,
            docstring="Test backbone strategy",
        )

        assert BackboneStrategy.update_backbone is True
        assert BackboneStrategy.update_neck is False
        assert BackboneStrategy.update_head is False

    def test_create_head_only_strategy(self):
        """Test creating a strategy that only updates head."""
        HeadStrategy = create_partial_strategy(
            fl.server.strategy.FedAvg,
            "TestFedHeadAvg",
            update_backbone=False,
            update_neck=False,
            update_head=True,
            docstring="Test head strategy",
        )

        assert HeadStrategy.update_backbone is False
        assert HeadStrategy.update_neck is False
        assert HeadStrategy.update_head is True


class TestStrategyClasses:
    """Tests for pre-defined strategy classes."""

    def test_fedavg_updates_all_sections(self):
        """Test that FedAvg updates all model sections."""
        assert FedAvg.update_backbone is True
        assert FedAvg.update_neck is True
        assert FedAvg.update_head is True

    def test_fedbackboneavg_updates_only_backbone(self):
        """Test that FedBackboneAvg only updates backbone."""
        assert FedBackboneAvg.update_backbone is True
        assert FedBackboneAvg.update_neck is False
        assert FedBackboneAvg.update_head is False

    def test_fedneckavg_updates_only_neck(self):
        """Test that FedNeckAvg only updates neck."""
        assert FedNeckAvg.update_backbone is False
        assert FedNeckAvg.update_neck is True
        assert FedNeckAvg.update_head is False

    def test_fedheadavg_updates_only_head(self):
        """Test that FedHeadAvg only updates head."""
        assert FedHeadAvg.update_backbone is False
        assert FedHeadAvg.update_neck is False
        assert FedHeadAvg.update_head is True

    def test_fedbackboneneckavg_updates_backbone_and_neck(self):
        """Test that FedBackboneNeckAvg updates backbone and neck."""
        assert FedBackboneNeckAvg.update_backbone is True
        assert FedBackboneNeckAvg.update_neck is True
        assert FedBackboneNeckAvg.update_head is False
