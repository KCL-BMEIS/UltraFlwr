# Dataset Format

UltraFlwr uses the YOLO-compatible dataset format for all supported tasks. This document describes the required directory structure, annotation formats, and configuration files.

## Overview

UltraFlwr supports four vision tasks, each using Ultralytics YOLO-compatible format:
- **Object Detection** (`detect`)
- **Instance Segmentation** (`segment`)
- **Pose Estimation** (`pose`)
- **Image Classification** (`classify`)

All datasets follow a consistent structure with federated partitioning support.

## Directory Structure

### Global Dataset Structure

```
datasets/
└── {dataset_name}/
    ├── data.yaml              # Global dataset configuration
    ├── train/                 # Training split (before partitioning)
    │   ├── images/           # Training images
    │   └── labels/           # Training annotations
    ├── valid/                # Validation split (before partitioning)
    │   ├── images/
    │   └── labels/
    ├── test/                 # Test split (before partitioning)
    │   ├── images/
    │   └── labels/
    └── partitions/           # Client-specific data splits (created by fedyolo-partition)
        ├── client_0/
        │   ├── data.yaml     # Client-specific configuration
        │   ├── train/
        │   │   ├── images/
        │   │   └── labels/
        │   ├── valid/
        │   │   ├── images/
        │   │   └── labels/
        │   └── test/
        │       ├── images/
        │       └── labels/
        └── client_1/
            └── ...
```

### Classification Dataset Structure

Classification datasets have a different structure where images are organized into class folders:

```
datasets/
└── {dataset_name}/
    └── partitions/
        └── client_0/
            ├── data.yaml
            ├── train/
            │   ├── 0/        # Class 0 images
            │   ├── 1/        # Class 1 images
            │   └── ...
            ├── test/
            │   ├── 0/
            │   ├── 1/
            │   └── ...
            └── train.cache   # Ultralytics cache file
```

## Configuration File: data.yaml

### Global data.yaml

Located at `datasets/{dataset_name}/data.yaml`, this file defines the dataset structure:

```yaml
# Path configuration (relative to data.yaml location)
train: ../train/images
val: ../valid/images
test: ../test/images

# Class information
nc: 2                         # Number of classes
names: ['class1', 'class2']   # Class names (order matters)

# Optional: Dataset metadata
roboflow:
  workspace: your-workspace
  project: your-project
  version: 1
  license: CC BY 4.0
  url: https://example.com/dataset
```

### Client-specific data.yaml

Located at `datasets/{dataset_name}/partitions/client_X/data.yaml`:

```yaml
# Paths relative to client directory
train: ./train/images
val: ./valid/images
test: ./test/images

# Class information (same as global)
nc: 2
names: ['class1', 'class2']

# Optional: Same metadata as global
roboflow:
  ...
```

## Annotation Formats

### Object Detection

Each image has a corresponding `.txt` file in the `labels/` directory with the same base name.

**Format:** One line per object
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer class index (0-indexed)
- `x_center`, `y_center`: Normalized coordinates of bounding box center (0-1)
- `width`, `height`: Normalized bounding box dimensions (0-1)

**Example:** `image1.txt`
```
1 0.633594 0.548438 0.050000 0.069531
1 0.695312 0.544531 0.037500 0.058594
0 0.438281 0.431250 0.026562 0.053906
```

### Instance Segmentation

Same format as detection, but with additional polygon points:

**Format:**
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

- `class_id`: Integer class index
- `x1 y1 x2 y2 ...`: Normalized polygon vertices defining the instance mask

**Example:** `image1.txt`
```
0 0.1 0.2 0.3 0.4 0.5 0.4 0.3 0.2
1 0.6 0.7 0.8 0.7 0.9 0.8 0.7 0.9
```

### Pose Estimation

Similar to detection with keypoint coordinates:

**Format:**
```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_visible> <kp2_x> <kp2_y> <kp2_visible> ...
```

- First 5 values: Same as detection format
- Remaining values: Keypoint triplets (x, y, visibility)
  - `kp_x`, `kp_y`: Normalized keypoint coordinates
  - `kp_visible`: 0 (not labeled), 1 (labeled but not visible), 2 (labeled and visible)

**Example:** `image1.txt`
```
0 0.5 0.5 0.3 0.4 0.45 0.35 2 0.55 0.35 2 0.5 0.6 1
```

### Image Classification

No label files needed. Images are organized into class-named subdirectories:

```
train/
├── class_0/
│   ├── image1.jpg
│   └── image2.jpg
├── class_1/
│   ├── image3.jpg
│   └── image4.jpg
└── ...
```

The directory name determines the class label.

## Federated Partitioning

### Automatic Partitioning

UltraFlwr automatically partitions datasets across clients using the `fedyolo-partition` command:

```bash
fedyolo-partition
```

This command:
1. Reads the global dataset from `datasets/{dataset_name}/`
2. Splits data according to client ratios defined in `fedyolo/config.py`
3. Creates `partitions/client_X/` directories
4. Copies images and labels to each client partition
5. Generates client-specific `data.yaml` files

### Configuration

Client partitioning is configured in `fedyolo/config.py`:

```python
# Define number of clients per dataset
DETECTION_CLIENTS = {"baseline": 2}      # 2 clients for baseline dataset
SEGMENTATION_CLIENTS = {"seg": 1}        # 1 client for segmentation
POSE_CLIENTS = {"pose": 1}               # 1 client for pose
CLASSIFICATION_CLIENTS = {"mnist": 1}    # 1 client for classification
```

### Partition Distribution

By default, data is split equally across clients. For custom ratios, modify `CLIENT_RATIOS` in `fedyolo/config.py`:

```python
# Example: 70-30 split for 2 clients
CLIENT_RATIOS = [0.7, 0.3]
```

The partitioner:
- Splits train/valid/test independently
- Maintains class distribution (approximately equal per client)
- Prints class distribution tables for verification

## Image Requirements

### Supported Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)

### Recommendations

- **Resolution:** No strict requirements; YOLO auto-resizes during training
- **Naming:** Image and label files must have matching base names
  - Image: `image001.jpg`
  - Label: `image001.txt`
- **Organization:** Keep images and labels in separate directories

## Converting from Other Formats

UltraFlwr provides converters in `fedyolo/data/converters/`:

### COCO to YOLO (Example)

```python
from fedyolo.data.converters.endoscapes_coco_to_yolo import create_yolo_structure

data_home = "/path/to/coco/dataset"  # Contains train/, val/, test/ with annotation_coco.json
output_dir = "/path/to/output"       # Output YOLO format directory

create_yolo_structure(data_home, output_dir)
```

This converter:
- Reads COCO JSON annotations
- Converts bounding boxes to YOLO format
- Creates directory structure
- Generates `data.yaml`
- Reports statistics (frames with/without annotations, duplicates)

### Custom Converters

To convert from other formats:

1. Create images directory structure (`train/images/`, `valid/images/`, `test/images/`)
2. Convert annotations to YOLO format (normalized coordinates)
3. Create corresponding label files in `train/labels/`, `valid/labels/`, `test/labels/`
4. Generate `data.yaml` with class information

## SSL (Self-Supervised Learning) Compatibility

UltraFlwr's SSL training (`fedyolo-train-ssl`) uses the same dataset structure:

- SSL training uses images only (labels ignored)
- No format changes required
- Works with existing YOLO datasets
- Supports all task types (detection, segmentation, pose, classification)

The SSL phase extracts visual features from raw images, then supervised training uses labels for task-specific learning.

## Reference Examples

### Detection Dataset: baseline
```
datasets/baseline/
├── data.yaml (nc: 2, names: ['cheetah', 'human'])
├── train/images/ (thermal images)
├── train/labels/ (bounding boxes)
└── partitions/client_0/, client_1/
```

### Segmentation Dataset: seg
```
datasets/seg/
├── data.yaml (nc: N)
├── train/images/
├── train/labels/ (polygon segmentation masks)
└── partitions/client_0/
```

### Pose Dataset: pose
```
datasets/pose/
├── data.yaml (nc: 1)
├── train/images/
├── train/labels/ (keypoint annotations)
└── partitions/client_2/ (pose uses client_2 by default)
```

### Classification Dataset: mnist
```
datasets/mnist/
└── partitions/client_0/
    ├── data.yaml
    ├── train/0/, train/1/, ..., train/9/
    └── test/0/, test/1/, ..., test/9/
```

## Additional Resources

- [Ultralytics YOLO Datasets Documentation](https://docs.ultralytics.com/datasets/)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
- [UltraFlwr Partitioning](../fedyolo/data/partitioner.py)
- [Data Converters](../fedyolo/data/converters/)
