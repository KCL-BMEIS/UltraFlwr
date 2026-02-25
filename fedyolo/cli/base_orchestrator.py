#!/usr/bin/env python3
"""
Base Orchestrator for Flower Federated Learning.

This module provides the base class for orchestrating Flower federated learning
processes, with common functionality for managing supernodes, cleanup, and monitoring.
"""

import sys
import os
import time
import signal
import subprocess
import logging
from abc import ABC, abstractmethod


class BaseFlowerOrchestrator(ABC):
    """Base class for orchestrating Flower federated learning processes."""

    def __init__(self, mode_name="federated learning"):
        """
        Initialize the orchestrator.

        Args:
            mode_name: Human-readable name for the mode (e.g., "federated learning", "SSL training")
        """
        self.processes = []
        self.running = True
        self.mode_name = mode_name
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Shutting down {self.mode_name}...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Kill all child processes."""
        logger = logging.getLogger(__name__)

        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Process {proc.pid} did not terminate gracefully, forcing kill"
                )
                try:
                    proc.kill()
                    logger.info(f"Forcefully killed process {proc.pid}")
                except ProcessLookupError:
                    logger.debug(f"Process {proc.pid} already terminated")
                except Exception as e:
                    logger.error(f"Error killing process {proc.pid}: {e}")
            except Exception as e:
                logger.error(f"Error terminating process {proc.pid}: {e}")
        self.processes.clear()

    def free_port(self, port):
        """Free a port by killing processes using it."""
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"], capture_output=True, text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"  Freed port {port} (killed PID {pid})")
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
        except FileNotFoundError:
            # lsof not available, skip
            pass

    def start_superlink(self):
        """Start Flower superlink."""
        print("\n🌐 Starting Flower superlink...")

        # Free ports
        for port in [9091, 9092, 9093]:
            self.free_port(port)

        # Start superlink
        proc = subprocess.Popen(
            ["flower-superlink", "--insecure"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.processes.append(proc)
        print(f"  ✓ Superlink started (PID: {proc.pid})")
        time.sleep(2)  # Wait for superlink to be ready

    def monitor_training(self, log_file, completion_message="Training completed"):
        """
        Monitor training log for completion.

        Args:
            log_file: Path to the log file to monitor
            completion_message: Message to display when training completes
        """
        print("\n📊 Monitoring training progress...")
        print(f"   Watching: {log_file}")
        print("   Press Ctrl+C to stop\n")

        # Tail the log file
        try:
            with open(log_file, "r") as f:
                # Seek to end
                f.seek(0, 2)

                while self.running:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                        if "finished" in line.lower():
                            print(f"\n✅ {completion_message}!")
                            time.sleep(2)
                            return
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⚠️  Training monitoring interrupted by user")
        except IOError as e:
            print(f"\n❌ Error reading log file: {e}")

    @abstractmethod
    def start_supernode(self, client_id):
        """
        Start a Flower supernode for a client.

        Args:
            client_id: The client ID

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def start_serverapp(self):
        """
        Start the Flower ServerApp.

        Must be implemented by subclasses.
        """
        pass

    def run(self):
        """
        Run the complete federated learning workflow.

        This method can be overridden by subclasses if needed.
        """
        try:
            # Start superlink
            self.start_superlink()

            # Start supernodes for all clients
            from fedyolo.config import CLIENT_CONFIG

            for client_id in CLIENT_CONFIG.keys():
                self.start_supernode(client_id)

            # Small delay to ensure all supernodes are ready
            time.sleep(2)

            # Start ServerApp (this will block until training completes)
            self.start_serverapp()

        finally:
            self.cleanup()
