#!/usr/bin/env python3
"""
Federated testing CLI.

This script runs comprehensive federated evaluation across all clients and scoring styles.
Can also run specific tests when arguments are provided.
"""

import sys
import argparse
from fedyolo.test.evaluator import (
    client_client_metrics,
    client_server_metrics,
    server_client_metrics,
    server_server_metrics,
)
from fedyolo.config import SERVER_CONFIG, CLIENT_CONFIG, get_output_dirs


def run_all_tests():
    """Run comprehensive testing across all clients and scoring styles."""
    print("=" * 70)
    print("🧪 UltraFlwr Comprehensive Testing")
    print("=" * 70)

    strategy_name = SERVER_CONFIG["strategy"]
    num_rounds = SERVER_CONFIG["rounds"]
    num_clients = len(CLIENT_CONFIG)

    print(f"\nStrategy: {strategy_name}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients: {num_clients}")

    # Check if all clients have the same task
    first_task = CLIENT_CONFIG[0]["task"]
    all_tasks_same = all(
        CLIENT_CONFIG[i]["task"] == first_task for i in CLIENT_CONFIG.keys()
    )

    if all_tasks_same:
        print(f"All clients have the same task: {first_task}")
        print("\nRunning all scoring styles for all clients...")
    else:
        print("Clients have different tasks (multi-task FL)")
        print("\nRunning client-client tests only (models on own data)...")
        print("(Other scoring styles require same task across clients)")

    print("=" * 70)

    output_dirs = get_output_dirs()

    # Client-dependent tests
    for client_id in CLIENT_CONFIG.keys():
        client_config = CLIENT_CONFIG[client_id]
        dataset_name = client_config["dataset_name"]
        task = client_config["task"]
        data_path = client_config["data_path"]

        print(f"\n{'='*70}")
        print(f"📊 Testing Client {client_id} ({dataset_name}, {task})")
        print(f"{'='*70}")

        # 1. Client-Client: Model on own data (always run)
        print(f"\nClient-Client: Client {client_id} model on own data")
        try:
            client_client_metrics(
                client_id,
                dataset_name,
                strategy_name,
                task,
                data_path,
                data_source_client=None,
                output_dirs=output_dirs,
            )
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed: {e}")

        # Only run other scoring styles if all clients have same task
        if all_tasks_same:
            # 2. Client-Server: Model on server data
            print(f"\nClient-Server: Client {client_id} model on server data")
            try:
                client_server_metrics(
                    client_id,
                    dataset_name,
                    strategy_name,
                    task,
                    output_dirs=output_dirs,
                )
                print("✓ Success")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # 3. Server-Client: Server model on client data
            print(f"\nServer-Client: Server model on client {client_id} data")
            try:
                server_client_metrics(
                    client_id,
                    dataset_name,
                    strategy_name,
                    num_rounds,
                    task,
                    data_path,
                    output_dirs=output_dirs,
                )
                print("✓ Success")
            except Exception as e:
                print(f"✗ Failed: {e}")

    # 4. Server-Server: Server model on server data (only if all tasks same)
    if all_tasks_same:
        first_client = CLIENT_CONFIG[0]
        dataset_name = first_client["dataset_name"]
        task = first_client["task"]

        print(f"\n{'='*70}")
        print("Server-Server: Server model on server data")
        print(f"{'='*70}")
        try:
            server_server_metrics(
                dataset_name, strategy_name, num_rounds, task, output_dirs=output_dirs
            )
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed: {e}")

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print(f"Results saved to: {output_dirs['results_metrics']}")
    print("=" * 70)


def run_single_test(args):
    """Run a specific test based on arguments."""
    strategy_name = args.strategy_name or SERVER_CONFIG["strategy"]
    num_rounds = SERVER_CONFIG["rounds"]
    output_dirs = get_output_dirs()

    print(f"\n=== Running {args.scoring_style} test ===")
    print(f"Client: {args.client_num}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Strategy: {strategy_name}")
    print(f"Task: {args.task}")
    print("=" * 50 + "\n")

    try:
        if args.scoring_style == "client-client":
            # Get data path from config if not provided
            data_path = args.data_path
            if data_path is None and args.client_num < len(CLIENT_CONFIG):
                data_path = CLIENT_CONFIG[args.client_num]["data_path"]

            client_client_metrics(
                args.client_num,
                args.dataset_name,
                strategy_name,
                args.task,
                data_path,
                args.data_source_client,
                output_dirs,
            )

        elif args.scoring_style == "client-server":
            client_server_metrics(
                args.client_num,
                args.dataset_name,
                strategy_name,
                args.task,
                output_dirs,
            )

        elif args.scoring_style == "server-client":
            data_path = args.data_path
            if data_path is None and args.client_num < len(CLIENT_CONFIG):
                data_path = CLIENT_CONFIG[args.client_num]["data_path"]

            server_client_metrics(
                args.client_num,
                args.dataset_name,
                strategy_name,
                num_rounds,
                args.task,
                data_path,
                output_dirs,
            )

        elif args.scoring_style == "server-server":
            server_server_metrics(
                args.dataset_name, strategy_name, num_rounds, args.task, output_dirs
            )

        print("\n✓ Testing complete!")
        return 0

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main entry point for federated testing."""
    parser = argparse.ArgumentParser(
        description="Federated YOLO Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for all clients (comprehensive)
  fedyolo-test

  # Run specific test
  fedyolo-test --scoring-style client-client --client-num 0 --dataset-name baseline --task detect

  # Test server model on server data
  fedyolo-test --scoring-style server-server --dataset-name baseline --task detect
        """,
    )
    parser.add_argument(
        "--scoring-style",
        type=str,
        choices=["client-client", "client-server", "server-client", "server-server"],
        help="Scoring style (if not specified, runs all tests)",
    )
    parser.add_argument("--client-num", type=int, help="Client number to test")
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name (e.g., 'baseline', 'seg', 'pose')",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default=None,
        help="Strategy name (default: from SERVER_CONFIG)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "segment", "pose", "classify"],
        help="Task type (default: detect)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data.yaml (optional, inferred from config if not provided)",
    )
    parser.add_argument(
        "--data-source-client",
        type=int,
        default=None,
        help="Client ID whose data is being used (for cross-client testing)",
    )

    args = parser.parse_args()

    # If no arguments provided, run all tests
    if args.scoring_style is None:
        run_all_tests()
        return 0

    # Validate required arguments for single test
    if args.scoring_style != "server-server":
        if args.client_num is None:
            print("Error: --client-num is required for this scoring style")
            parser.print_help()
            return 1

    if args.dataset_name is None:
        # Try to infer from client config
        if args.client_num is not None and args.client_num < len(CLIENT_CONFIG):
            args.dataset_name = CLIENT_CONFIG[args.client_num]["dataset_name"]
            args.task = CLIENT_CONFIG[args.client_num]["task"]
            print(f"Inferred dataset: {args.dataset_name}, task: {args.task}")
        else:
            print("Error: --dataset-name is required")
            parser.print_help()
            return 1

    return run_single_test(args)


if __name__ == "__main__":
    sys.exit(main())
