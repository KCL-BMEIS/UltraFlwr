import unittest
from unittest.mock import MagicMock
import torch
from collections import OrderedDict
from FedYOLO.train.strategies import (
    FedAvg, FedHeadAvg, FedNeckAvg, FedBackboneAvg,
    FedNeckHeadAvg, FedBackboneHeadAvg, FedBackboneNeckAvg,
    FedMedian, FedHeadMedian, FedNeckMedian, FedBackboneMedian,
    FedNeckHeadMedian, FedBackboneHeadMedian, FedBackboneNeckMedian
)
from FedYOLO.train.server_utils import save_model_checkpoint
from FedYOLO.config import HOME, SPLITS_CONFIG

class TestStrategies(unittest.TestCase):

    def setUp(self):
        self.mock_parameters = MagicMock()
        self.mock_client_manager = MagicMock()
        self.mock_fit_results = [(MagicMock(), MagicMock())]
        self.mock_failures = []

        self.mock_state_dict = OrderedDict({
            "model.0.weight": torch.randn(3, 3),
            "model.9.weight": torch.randn(3, 3),
            "model.23.weight": torch.randn(3, 3),
        })

    def test_fedavg_strategy(self):
        strategy = FedAvg()
        strategy.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        strategy.update_backbone = True
        strategy.update_neck = True
        strategy.update_head = True

        strategy.load_and_update_model = MagicMock()
        strategy.load_and_update_model.side_effect = lambda aggregated_parameters: {
            k: v for k, v in self.mock_state_dict.items()
            if (strategy.update_backbone and k.startswith("model.0")) or
               (strategy.update_neck and k.startswith("model.9")) or
               (strategy.update_head and k.startswith("model.23"))
        }
        aggregated_parameters = strategy.load_and_update_model(self.mock_parameters)

        self.assertTrue("model.0.weight" in aggregated_parameters)
        self.assertTrue("model.9.weight" in aggregated_parameters)
        self.assertTrue("model.23.weight" in aggregated_parameters)

    def test_fedheadavg_strategy(self):
        strategy = FedHeadAvg()
        strategy.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        strategy.update_backbone = False
        strategy.update_neck = False
        strategy.update_head = True

        strategy.load_and_update_model = MagicMock()
        strategy.load_and_update_model.side_effect = lambda aggregated_parameters: {
            k: v for k, v in self.mock_state_dict.items()
            if (strategy.update_backbone and k.startswith("model.0")) or
               (strategy.update_neck and k.startswith("model.9")) or
               (strategy.update_head and k.startswith("model.23"))
        }
        aggregated_parameters = strategy.load_and_update_model(self.mock_parameters)

        self.assertFalse("model.0.weight" in aggregated_parameters)
        self.assertFalse("model.9.weight" in aggregated_parameters)
        self.assertTrue("model.23.weight" in aggregated_parameters)

    def test_fedneckavg_strategy(self):
        strategy = FedNeckAvg()
        strategy.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        strategy.update_backbone = False
        strategy.update_neck = True
        strategy.update_head = False

        strategy.load_and_update_model = MagicMock()
        strategy.load_and_update_model.side_effect = lambda aggregated_parameters: {
            k: v for k, v in self.mock_state_dict.items()
            if (strategy.update_backbone and k.startswith("model.0")) or
               (strategy.update_neck and k.startswith("model.9")) or
               (strategy.update_head and k.startswith("model.23"))
        }
        aggregated_parameters = strategy.load_and_update_model(self.mock_parameters)

        self.assertFalse("model.0.weight" in aggregated_parameters)
        self.assertTrue("model.9.weight" in aggregated_parameters)
        self.assertFalse("model.23.weight" in aggregated_parameters)

    def set_mock_side_effect(self, strategy):
        strategy.load_and_update_model = MagicMock()
        strategy.load_and_update_model.side_effect = lambda aggregated_parameters: {
            k: v for k, v in self.mock_state_dict.items()
            if (strategy.update_backbone and k.startswith("model.0")) or
               (strategy.update_neck and k.startswith("model.9")) or
               (strategy.update_head and k.startswith("model.23"))
        }

    def test_fedbackboneheadavg_strategy(self):
        strategy = FedBackboneHeadAvg()
        strategy.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        strategy.update_backbone = True
        strategy.update_neck = False
        strategy.update_head = True

        self.set_mock_side_effect(strategy)
        aggregated_parameters = strategy.load_and_update_model(self.mock_parameters)

        self.assertTrue("model.0.weight" in aggregated_parameters)
        self.assertFalse("model.9.weight" in aggregated_parameters)
        self.assertTrue("model.23.weight" in aggregated_parameters)

    def test_fedneckheadavg_strategy(self):
        strategy = FedNeckHeadAvg()
        strategy.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        strategy.update_backbone = False
        strategy.update_neck = True
        strategy.update_head = True

        self.set_mock_side_effect(strategy)
        aggregated_parameters = strategy.load_and_update_model(self.mock_parameters)

        self.assertFalse("model.0.weight" in aggregated_parameters)
        self.assertTrue("model.9.weight" in aggregated_parameters)
        self.assertTrue("model.23.weight" in aggregated_parameters)

    def test_fedbackboneneckmedian_strategy(self):
        strategy = FedBackboneNeckMedian()
        strategy.model_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{SPLITS_CONFIG['dataset_name']}.yaml"
        strategy.update_backbone = True
        strategy.update_neck = True
        strategy.update_head = False

        self.set_mock_side_effect(strategy)
        aggregated_parameters = strategy.load_and_update_model(self.mock_parameters)

        self.assertTrue("model.0.weight" in aggregated_parameters)
        self.assertTrue("model.9.weight" in aggregated_parameters)
        self.assertFalse("model.23.weight" in aggregated_parameters)

if __name__ == "__main__":
    unittest.main()