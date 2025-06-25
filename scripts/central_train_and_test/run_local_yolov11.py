#!/usr/bin/env python

import os
import sys
import time
import subprocess

# Optional: Free port 8080
try:
    import psutil
except ImportError:
    psutil = None

def free_port(port):
    if psutil is None:
        print("psutil not installed, using netstat/taskkill to free port", port)
        try:
            netstat = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True, text=True
            )
            pids = {line.strip().split()[-1] for line in netstat.splitlines()}
            for pid in pids:
                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                print(f"Killed PID {pid} on port {port}")
        except subprocess.CalledProcessError:
            pass
    else:
        for conn in psutil.net_connections():
            if conn.laddr and conn.laddr.port == port:
                try:
                    psutil.Process(conn.pid).kill()
                    print(f"Killed process {conn.pid} on port {port}")
                except Exception as e:
                    print(f"Could not kill PID {conn.pid}: {e}")

def main():
    # Step 1: Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    os.chdir(repo_root)
    print("CWD →", repo_root)
    sys.path.insert(0, repo_root)

    # Step 2: Load config
    from FedYOLO.config import SPLITS_CONFIG, SERVER_CONFIG, CLIENT_CONFIG

    dataset_path = SPLITS_CONFIG['dataset']
    dataset_name = SPLITS_CONFIG['dataset_name']
    partition_method = SPLITS_CONFIG['partition_method']  # should be "dirichlet"
    strategy_name = SERVER_CONFIG['strategy']             # e.g., "FedAvg"

    # Dynamic folder naming for logging/output
    run_folder_name = f"{strategy_name}_{dataset_name}_{partition_method}"

    # Actual folder where data will be saved
    partition_folder = f"partition_{partition_method}"
    partition_path = os.path.join(dataset_path, partition_folder)

    # Step 3: Run partitioning if not already done
    if not os.path.exists(partition_path):
        print(f" Partitioning not found at {partition_path}. Generating splits…")
        if partition_method == "fedssar":
            from FedYOLO.data_partitioner.fedssar_split import fedssar_split_dataset
            fedssar_split_dataset(SPLITS_CONFIG)
        elif partition_method == "dirichlet":
            from FedYOLO.data_partitioner.dirichlet_split import dirichlet_split_dataset
            dirichlet_split_dataset(SPLITS_CONFIG, alpha=0.5)  # You can change alpha if needed
        else:
            raise ValueError(f"Unknown partition method: {partition_method}")
        print("Partitioning complete.")

    # Step 4: Prepare directories
    logs_dir = os.path.join(repo_root, 'logs', run_folder_name)
    outputs_dir = os.path.join(repo_root, 'outputs', run_folder_name)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    server_script = os.path.join(repo_root, 'FedYOLO', 'train', 'yolo_server.py')
    client_script = os.path.join(repo_root, 'FedYOLO', 'train', 'yolo_client.py')

    # Step 5: Start server
    print("Freeing port 8080…")
    free_port(8080)

    server_log = os.path.join(logs_dir, "server_log.txt")
    print(" Starting server… logs →", server_log)
    procs = []
    with open(server_log, 'w') as f:
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root
        p = subprocess.Popen(
            [sys.executable, server_script],
            stdout=f, stderr=subprocess.STDOUT,
            env=env
        )
        procs.append(p)
        print("   Server PID:", p.pid)

    time.sleep(2)  # Let server boot up

    # Step 6: Start clients
    for cid in CLIENT_CONFIG:
        data_path = CLIENT_CONFIG[cid]['data_path']
        client_log = os.path.join(logs_dir, f"client_{cid}_log.txt")
        client_output_dir = os.path.join(outputs_dir, f"client_{cid}")
        os.makedirs(client_output_dir, exist_ok=True)

        print(f"👤 Starting client {cid}… log → {client_log}, output → {client_output_dir}")

        with open(client_log, 'w') as f:
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_root
            env["OUTPUT_DIR"] = client_output_dir
            p = subprocess.Popen(
                [
                    sys.executable, client_script,
                    f"--cid={cid}",
                    f"--data_path={data_path}"
                ],
                stdout=f, stderr=subprocess.STDOUT,
                env=env
            )
            procs.append(p)
            print(f"  Client {cid} PID:", p.pid)

    # Step 7: Wait for all processes to finish
    print("⏳ Waiting for all processes to complete…")
    for p in procs:
        p.wait()
    print("Training complete.")

if __name__ == "__main__":
    main()
