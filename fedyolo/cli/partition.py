#!/usr/bin/env python3
"""
Data partitioning CLI wrapper.

This script provides a convenient entry point for partitioning datasets
for federated learning.
"""

import sys
from fedyolo.data.partitioner import generate_splits_configs, split_dataset


def main():
    """Partition datasets for federated learning."""
    print("Partitioning datasets for federated learning...\n")

    # Generate split configs for each dataset
    splits_configs = generate_splits_configs()

    # Process each dataset
    for dataset_name, config in splits_configs.items():
        split_dataset(config)

    print("\n✓ Dataset partitioning complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
