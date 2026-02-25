[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square&logo=python&logoColor=white)

# UltraFlwr: Mixed Supervision Federated Multi-Task Learning

Official repository for federated multi-task learning with YOLO, featuring integrated self-supervised learning (SSL) for pretraining without labels.

UltraFlwr provides random and equally sized YOLO compatible data partitioning, federated training, and flexible testing capabilities across multiple vision tasks (detection, segmentation, pose estimation, classification). It integrates [Ultralytics](https://github.com/Ultralytics/Ultralytics) YOLO off-the-shelf within the [Flower](https://github.com/adap/flower) framework.

For more details on the motivation behind this project, see [docs/motivation.md](docs/motivation.md).

## Quick Start

```bash
# Install
uv pip install -e . # or just pip install -e .

# Partition datasets (one-time setup)
fedyolo-partition

# Phase 1: Federated SSL pretraining
fedyolo-train-ssl

# Phase 2: Supervised training (automatically uses SSL-pretrained weights)
fedyolo-train

# Run evaluation
fedyolo-test
```

See below for detailed usage guides and advanced options.

## Benchmarks

Comprehensive benchmarks are included in the [benchmarks](benchmarks) folder.

## Usage (Training)

We provide usage guides using [pills dataset](https://universe.roboflow.com/roboflow-100/pills-sxdht) under these settings:

1. [Single machine simulation using Python virtual environment](docs/local_venv.md)
2. [Single machine simulation using Docker](docs/local_docker.md)

## Usage (Testing)

For testing and getting client-wise global and local scores: `fedyolo-test`
- This automatically prints out tables in Ultralytics style and tests all clients with all scoring styles.

## Federated Self-Supervised Learning (SSL)

UltraFlwr now includes integrated federated self-supervised learning (SSL) using the [Lightly](https://github.com/lightly-ai/lightly) library. This enables pretraining of visual feature extractors without labeled data before supervised fine-tuning.

### Two-Phase Workflow

**Phase 1: SSL Pretraining**
```bash
fedyolo-train-ssl
```
- Runs federated SSL pretraining on raw images (no labels needed)
- Aggregates learned visual representations across clients
- Saves SSL-pretrained backbone weights

**Phase 2: Supervised Fine-Tuning**
```bash
fedyolo-train
```
- Automatically loads SSL-pretrained weights
- Trains task-specific heads with labeled data
- Achieves better performance with less labeled data

### SSL Methods Supported

- **BYOL** (recommended for federated learning)
- **SimCLR** (contrastive learning)
- **MoCo** (momentum contrast)
- **Barlow Twins**
- **VICReg**

### Key Features

- Task-agnostic SSL: Works with detection, segmentation, pose estimation, classification
- Heterogeneous SSL: Different SSL methods per client (experimental)
- No data changes: Uses existing YOLO dataset structure

For detailed SSL documentation, configuration options, and usage examples, see [docs/ssl.md](docs/ssl.md).
