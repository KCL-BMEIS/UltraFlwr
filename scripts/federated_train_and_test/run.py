#!/usr/bin/env python
"""
run_fedyolo.py

Windows‐compatible Python launcher for FedYOLO server + clients.
"""

import os
import sys
import time
import subprocess

# --- (optional) use psutil to kill the process on port 8080 ---
try:
    import psutil
except ImportError:
    psutil = None

def free_port(port):
    if psutil is None:
        # fallback: use netstat+taskkill
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
    # 1) cd to repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    os.chdir(repo_root)
    print("CWD →", repo_root)

    # 2) import config
    config_path = os.path.join(repo_root, 'FedYOLO', 'config.py')
    if not os.path.isfile(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    sys.path.insert(0, repo_root)
    from FedYOLO.config import SPLITS_CONFIG, SERVER_CONFIG, CLIENT_CONFIG

    dataset_name  = SPLITS_CONFIG['dataset_name']
    strategy_name = SERVER_CONFIG['strategy']

    # 3) prepare paths
    server_script = os.path.join(repo_root, 'FedYOLO', 'train', 'yolo_server.py')
    client_script = os.path.join(repo_root, 'FedYOLO', 'train', 'yolo_client.py')
    logs_dir      = os.path.join(repo_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    procs = []

    # 4) start server
    print("Freeing port 8080…")
    free_port(8080)
    server_log = os.path.join(logs_dir, f"server_log_{dataset_name}_{strategy_name}.txt")
    print("Starting server… logs →", server_log)
    with open(server_log, 'w') as f:
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root
        p = subprocess.Popen(
            [sys.executable, server_script],
            stdout=f, stderr=subprocess.STDOUT,
            env=env
        )
        procs.append(p)
        print("  PID:", p.pid)

    # 5) give server a moment
    time.sleep(2)

    # 6) start clients
    for cid in CLIENT_CONFIG:
        data_path = CLIENT_CONFIG[cid]['data_path']
        client_log = os.path.join(logs_dir,
         f"client_{cid}_log_{dataset_name}_{strategy_name}.txt"
)

        print(f"Starting client {cid}… data_path={data_path} → logs: {client_log}")
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
            print("  PID:", p.pid)

    # 7) wait for all
    print("Waiting for all processes to finish…")
    for p in procs:
        p.wait()
    print("All done.")

if __name__ == "__main__":
    main()
