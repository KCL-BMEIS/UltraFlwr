#!/usr/bin/env python3
"""
Federated training CLI.

This script handles the complete federated training workflow including:
- Initializing experiment directories
- Starting Flower superlink
- Launching supernodes for each client
- Running the ServerApp
- Managing process lifecycle
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from fedyolo.cli.base_orchestrator import BaseFlowerOrchestrator
from fedyolo.train.output_manager import initialize_experiment_output
from fedyolo.config import SERVER_CONFIG, CLIENT_CONFIG, get_output_dirs


class FlowerOrchestrator(BaseFlowerOrchestrator):
    """Orchestrates Flower federated learning processes."""

    def __init__(self):
        super().__init__(mode_name="federated learning")

    def start_supernode(self, client_id):
        """Start a Flower supernode for a client."""
        client_config = CLIENT_CONFIG[client_id]
        data_path = client_config["data_path"]
        dataset_name = client_config["dataset_name"]
        strategy_name = SERVER_CONFIG["strategy"]

        # Get output directories
        output_dirs = get_output_dirs()
        log_dir = Path(output_dirs["logs_clients"]) / f"client_{client_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"training_{dataset_name}_{strategy_name}.log"

        # Calculate port
        port = 9090 + client_id + 4

        # Free port
        self.free_port(port)

        print(f"\n👤 Starting supernode for Client {client_id}...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port}")
        print(f"  Log: {log_file}")

        # Start supernode
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                [
                    "flower-supernode",
                    "--insecure",
                    "--superlink",
                    "127.0.0.1:9092",
                    "--clientappio-api-address",
                    f"127.0.0.1:{port}",
                    "--node-config",
                    f'cid={client_id} data_path="{data_path}"',
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            self.processes.append(proc)
            print(f"  ✓ Supernode started (PID: {proc.pid})")

    def start_serverapp(self):
        """Start the Flower ServerApp."""
        output_dirs = get_output_dirs()
        strategy_name = SERVER_CONFIG["strategy"]

        log_dir = Path(output_dirs["logs_server"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"server_{strategy_name}.log"

        print("\n🖥️  Starting ServerApp...")
        print(f"  Strategy: {strategy_name}")
        print(f"  Rounds: {SERVER_CONFIG['rounds']}")
        print(f"  Log: {log_file}")

        # Start ServerApp
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                ["flwr", "run", ".", "local-deployment", "--stream"],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self.processes.append(proc)
            print(f"  ✓ ServerApp started (PID: {proc.pid})")

            # Monitor log for completion
            self.monitor_training(log_file, completion_message="Training completed")


def main():
    """Main entry point for federated training."""
    parser = argparse.ArgumentParser(
        description="Federated YOLO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run federated training with default settings
  fedyolo-train

  # Initialize directories only
  fedyolo-train --init-only

  # Run with custom experiment name
  fedyolo-train --experiment-name my_experiment
        """,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name of the experiment (default: strategy name from config)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize output directories, don't start training",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("🚀 UltraFlwr Federated Training")
    print("=" * 60)

    # Initialize experiment output structure
    print("\n📁 Initializing experiment directories...")
    output_dirs = initialize_experiment_output(
        experiment_name=args.experiment_name, save_config=True
    )

    print("\n✓ Experiment initialized:")
    print(f"  Output: {output_dirs['root']}")
    print(f"  Strategy: {SERVER_CONFIG['strategy']}")
    print(f"  Rounds: {SERVER_CONFIG['rounds']}")
    print(f"  Clients: {len(CLIENT_CONFIG)}")

    if args.init_only:
        print("\n✓ Initialization complete!")
        return 0

    # Run federated learning
    orchestrator = FlowerOrchestrator()
    orchestrator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
