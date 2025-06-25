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
    partition_method = SPLITS_CONFIG['partition_method']  # NEW
    strategy_name = SERVER_CONFIG['strategy']

    # Step 3: Run FedSSAR splitting if needed
    fedssar_path = os.path.join(dataset_path, "partition_fedssar")
    if not os.path.exists(fedssar_path) and partition_method == 'fedssar':
        print("⚙️ FedSSAR partitioning not found. Running OPTICS-based split…")
        from FedYOLO.data_partitioner.fedssar_split import fedssar_split_dataset
        fedssar_split_dataset(SPLITS_CONFIG)
        print("✅ FedSSAR split complete.")

    # Step 4: Prepare paths
    server_script = os.path.join(repo_root, 'FedYOLO', 'train', 'yolo_server.py')
    client_script = os.path.join(repo_root, 'FedYOLO', 'train', 'yolo_client.py')
    logs_dir = os.path.join(repo_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Step 5: Start server
    print("Freeing port 8080…")
    free_port(8080)

    # 🚀 Updated log file name to include partition method
    server_log = os.path.join(logs_dir, f"server_log_{dataset_name}_{partition_method}_{strategy_name}.txt")
    print("🚀 Starting server… logs →", server_log)
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
        print("  🧠 Server PID:", p.pid)

    time.sleep(2)  # Let the server initialize

    # Step 6: Start all clients
    for cid in CLIENT_CONFIG:
        data_path = CLIENT_CONFIG[cid]['data_path']

        # 👤 Updated client log filename
        client_log = os.path.join(logs_dir, f"client_{cid}_log_{dataset_name}_{partition_method}_{strategy_name}.txt")
        print(f"👤 Starting client {cid}… → logs: {client_log}")

        # 🔽 Optional: Save results in unique client output folders like: FedAvg_bccd_fedssar_0
        output_dir = os.path.join(repo_root, f"{strategy_name}_{dataset_name}_{partition_method}_{cid}")
        os.makedirs(output_dir, exist_ok=True)  # Can be used inside your client script

        with open(client_log, 'w') as f:
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_root
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
            print(f"  ✅ Client {cid} PID:", p.pid)

    # Step 7: Wait for all
    print("⏳ Waiting for all processes to complete…")
    for p in procs:
        p.wait()
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
