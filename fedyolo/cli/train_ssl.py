#!/usr/bin/env python3
"""
Federated SSL (Self-Supervised Learning) Training CLI.

This script handles federated SSL pretraining workflow including:
- Validating SSL configuration
- Initializing experiment directories
- Starting Flower superlink
- Launching supernodes for each client (SSL mode)
- Running the ServerApp for SSL aggregation
- Saving SSL-pretrained weights
"""

import sys
import os

# IMPORTANT: Set SSL mode BEFORE any fedyolo imports
# This ensures that config.py and other modules see the SSL mode flag
os.environ["FEDYOLO_SSL_MODE"] = "true"

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from fedyolo.cli.base_orchestrator import BaseFlowerOrchestrator
from fedyolo.train.output_manager import initialize_experiment_output
from fedyolo.config import (
    SSL_SERVER_CONFIG,
    SSL_CONFIG,
    CLIENT_CONFIG,
    get_output_dirs,
)
from fedyolo.train.ssl_client import validate_ssl_config
from fedyolo.train.output_manager import ensure_weights_available


class SSLFlowerOrchestrator(BaseFlowerOrchestrator):
    """Orchestrates Flower federated SSL learning processes."""

    def __init__(self):
        super().__init__(mode_name="federated SSL training")

    def start_supernode(self, client_id):
        """Start a Flower supernode for a client (SSL mode)."""
        client_config = CLIENT_CONFIG[client_id]
        data_path = client_config["data_path"]
        dataset_name = client_config["dataset_name"]
        task = client_config["task"]

        # For SSL, we use SSL-specific experiment name
        output_dirs = get_output_dirs("ssl_pretraining")
        log_dir = Path(output_dirs["logs_clients"]) / f"client_{client_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"ssl_{dataset_name}_{SSL_CONFIG['method']}.log"

        # Calculate port
        port = 9090 + client_id + 4

        # Free port
        self.free_port(port)

        print(f"\n👤 Starting SSL supernode for Client {client_id}...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Task: {task}")
        print(f"  SSL Method: {SSL_CONFIG.get('method', 'byol')}")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port}")
        print(f"  Log: {log_file}")

        # Start supernode with SSL mode enabled via environment variable
        ssl_env = os.environ.copy()
        ssl_env["FEDYOLO_SSL_MODE"] = "true"

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
                env=ssl_env,  # Pass environment with SSL mode enabled
            )
            self.processes.append(proc)
            print(f"  ✓ SSL Supernode started (PID: {proc.pid})")

    def start_serverapp(self):
        """Start the Flower ServerApp for SSL aggregation."""
        output_dirs = get_output_dirs("ssl_pretraining")
        strategy_name = SSL_SERVER_CONFIG["strategy"]

        log_dir = Path(output_dirs["logs_server"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"ssl_server_{strategy_name}.log"

        print("\n🖥️  Starting SSL ServerApp...")
        print(f"  Strategy: {strategy_name}")
        print(f"  Rounds: {SSL_SERVER_CONFIG['rounds']}")
        print(f"  SSL Method: {SSL_CONFIG.get('method', 'byol')}")
        print(f"  Log: {log_file}")

        # Start ServerApp with SSL mode enabled
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                ["flwr", "run", ".", "local-deployment", "--stream"],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "FEDYOLO_SSL_MODE": "true"},
            )
            self.processes.append(proc)
            print(f"  ✓ SSL ServerApp started (PID: {proc.pid})")

            # Monitor log for completion
            self.monitor_training(log_file, completion_message="SSL Training completed")

    def save_ssl_weights(self):
        """
        Save SSL-pretrained weights after training completes.

        Attempts to save the aggregated SSL backbone weights from the server.
        Falls back to copying base weights if server weights not available.
        """
        print("\n💾 Saving SSL-pretrained weights...")

        # Create SSL weights directory
        ssl_weights_dir = Path(SSL_CONFIG["save_path"])
        ssl_weights_dir.mkdir(parents=True, exist_ok=True)
        ssl_weight_file = ssl_weights_dir / "yolo11n-ssl.pt"

        # Try to get the final server checkpoint from SSL training
        output_dirs = get_output_dirs("ssl_pretraining")
        server_checkpoint_dir = Path(output_dirs["checkpoints_server"])

        # Look for the final round checkpoint
        final_round = SSL_SERVER_CONFIG.get("rounds", 5)
        checkpoint_pattern = f"round_{final_round}_*_*.pt"

        checkpoints = list(server_checkpoint_dir.glob(checkpoint_pattern))

        if checkpoints:
            # Found server checkpoint - use it as SSL weights
            latest_checkpoint = checkpoints[0]
            print(f"  Found server checkpoint: {latest_checkpoint.name}")

            # Copy server checkpoint to SSL weights location
            shutil.copy2(latest_checkpoint, ssl_weight_file)
            print(f"  ✓ Copied server checkpoint to: {ssl_weight_file}")
        else:
            # No server checkpoint found - create from base weights
            print("  ⚠️  No server checkpoint found, using base weights as fallback")

            base_weights_dir = get_output_dirs()["weights_base"]

            # Ensure weights are in weights/base/, download if needed
            base_weight_path = ensure_weights_available("yolo11n.pt", base_weights_dir)

            # Copy base weights to SSL location
            shutil.copy2(base_weight_path, ssl_weight_file)
            print(f"  ✓ Copied base weights to: {ssl_weight_file}")
            print("  ⚠️  Note: Using base weights, not SSL-trained weights")

        # Save SSL configuration metadata
        metadata_file = ssl_weights_dir / "ssl_metadata.json"
        metadata = {
            "method": SSL_CONFIG.get("method"),
            "rounds": SSL_SERVER_CONFIG.get("rounds"),
            "ssl_epochs": SSL_CONFIG.get("ssl_epochs"),
            "batch_size": SSL_CONFIG.get("ssl_batch_size"),
            "clients": len(CLIENT_CONFIG),
            "weights_source": "server_checkpoint" if checkpoints else "base_weights",
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Metadata saved: {metadata_file}")
        print("  ✓ SSL weights ready for supervised training!")

    def run(self):
        """Run the complete federated SSL learning workflow."""
        try:
            # Use base class run() which handles superlink, supernodes, and serverapp
            super().run()

            # Save SSL weights after training completes
            self.save_ssl_weights()

        except Exception:
            # Base class already called cleanup in finally block
            raise


def main():
    """Main entry point for federated SSL training."""
    parser = argparse.ArgumentParser(
        description="Federated Self-Supervised Learning (SSL) for YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SSL pretraining with default settings (BYOL)
  fedyolo-train-ssl

  # Run with specific SSL method
  fedyolo-train-ssl --method simclr

  # Run with custom number of rounds and epochs
  fedyolo-train-ssl --rounds 10 --ssl-epochs 30

  # Allow heterogeneous SSL methods (experimental)
  fedyolo-train-ssl --allow-heterogeneous

After SSL training completes, run supervised training:
  fedyolo-train
        """,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["byol", "simclr", "moco", "barlow_twins", "vicreg"],
        default=None,
        help="SSL method to use (default: from config)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of federated rounds (default: from config)",
    )
    parser.add_argument(
        "--ssl-epochs",
        type=int,
        default=None,
        help="Number of SSL epochs per round (default: from config)",
    )
    parser.add_argument(
        "--allow-heterogeneous",
        action="store_true",
        help="Allow different SSL methods per client (experimental)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize output directories, don't start training",
    )

    args = parser.parse_args()

    # Update SSL config with CLI arguments
    if args.method:
        SSL_CONFIG["method"] = args.method
    if args.rounds:
        SSL_SERVER_CONFIG["rounds"] = args.rounds
    if args.ssl_epochs:
        SSL_CONFIG["ssl_epochs"] = args.ssl_epochs
    if args.allow_heterogeneous:
        SSL_CONFIG["allow_heterogeneous"] = True

    # Print header
    print("=" * 60)
    print("🧠 UltraFlwr Federated SSL Training")
    print("=" * 60)

    # Validate SSL configuration
    print("\n🔍 Validating SSL configuration...")
    try:
        validate_ssl_config()
        print("  ✓ SSL configuration validated")
    except ValueError as e:
        print(f"\n❌ Configuration error:\n{e}")
        return 1

    # Print SSL configuration
    print("\n📋 SSL Configuration:")
    print(f"  Method: {SSL_CONFIG.get('method', 'byol')}")
    print(f"  Rounds: {SSL_SERVER_CONFIG.get('rounds', 5)}")
    print(f"  Epochs per round: {SSL_CONFIG.get('ssl_epochs', 20)}")
    print(f"  Batch size: {SSL_CONFIG.get('ssl_batch_size', 64)}")
    print(f"  Clients: {len(CLIENT_CONFIG)}")
    print(f"  Strategy: {SSL_SERVER_CONFIG.get('strategy', 'FedBackboneAvg')}")

    # Initialize experiment output structure
    print("\n📁 Initializing SSL experiment directories...")
    output_dirs = initialize_experiment_output(
        experiment_name="ssl_pretraining", save_config=True
    )

    print("\n✓ SSL Experiment initialized:")
    print(f"  Output: {output_dirs['root']}")
    print(f"  SSL weights: {SSL_CONFIG['save_path']}")

    if args.init_only:
        print("\n✓ Initialization complete!")
        return 0

    # Run federated SSL learning
    orchestrator = SSLFlowerOrchestrator()
    orchestrator.run()

    print("\n" + "=" * 60)
    print("✅ SSL Pretraining Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run supervised training: fedyolo-train")
    print("  2. Supervised training will automatically use SSL weights")

    return 0


if __name__ == "__main__":
    sys.exit(main())
