"""Server helper: execute commands on remote server via SSH (paramiko).

Usage:
  python server_helper.py <command>
  python server_helper.py setup        # git clone + pip install
  python server_helper.py train        # start full benchmark
  python server_helper.py status       # check training status
  python server_helper.py gpu          # check GPU status
"""

import os
import sys
import time

import paramiko

HOST = "hz-4.matpool.com"
PORT = 27301
USER = "root"
PASSWORD = "IP#T=jXj#ol[Fs1d"
REPO_URL = "git@github.com:j1x3l0/Dyna-Q-DQN-UAV.git"
REPO_DIR = "/root/Dyna-Q-DQN-UAV"

TRAINING_CMD = (
    "cd {repo} && "
    "nohup python scripts/run_full_benchmark.py "
    "--algos iddpg,maddpg,nodyna,dyna "
    "--seeds 42,123,2026,7,2023 "
    "--case 1 "
    "> benchmark_$(date +%Y%m%d_%H%M%S).log 2>&1 & "
    "echo PID=$!"
)


def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    return client


def run_cmd(client, cmd, timeout=120):
    print(f"\n>>> {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    if out.strip():
        print(out.strip())
    if err.strip():
        print(f"[stderr] {err.strip()}")
    return out, err


def cmd_setup(client):
    """Clone repo and install dependencies."""
    # Check if repo already exists
    out, _ = run_cmd(client, f"test -d {REPO_DIR} && echo EXISTS || echo NOT_FOUND")
    if "NOT_FOUND" in out:
        run_cmd(client, f"git clone {REPO_URL} {REPO_DIR}", timeout=120)
    else:
        print("Repo exists, pulling latest...")
        run_cmd(client, f"cd {REPO_DIR} && git pull origin main", timeout=60)

    # Check Python and install dependencies
    run_cmd(client, "which python3 && python3 --version")
    run_cmd(client, "which pip3 && pip3 --version")
    run_cmd(client, f"cd {REPO_DIR} && pip3 install -r requirements.txt", timeout=120)
    # Ensure PyTorch with CUDA is installed
    run_cmd(client, "python3 -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'",
             timeout=60)


def cmd_train(client):
    """Start the full benchmark training in background."""
    cmd = TRAINING_CMD.format(repo=REPO_DIR)
    run_cmd(client, cmd, timeout=10)


def cmd_status(client):
    """Check training status."""
    run_cmd(client, f"ps aux | grep run_full_benchmark | grep -v grep")
    run_cmd(client, f"ls -lt {REPO_DIR}/results/ | head -10")
    run_cmd(client, f"ls -lt {REPO_DIR}/checkpoints/ | head -10")
    # Tail the latest log
    run_cmd(client, f"ls -t {REPO_DIR}/*.log 2>/dev/null | head -1 | xargs tail -20")


def cmd_gpu(client):
    """Check GPU status."""
    run_cmd(client, "nvidia-smi 2>/dev/null || echo 'nvidia-smi not found'")
    run_cmd(client, "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")' 2>/dev/null || echo 'torch not available'")


def cmd_log(client):
    """Tail the latest training log."""
    run_cmd(client, f"ls -t {REPO_DIR}/benchmark_*.log 2>/dev/null | head -1 | xargs tail -50")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    client = ssh_connect()
    try:
        if action == "setup":
            cmd_setup(client)
        elif action == "train":
            cmd_train(client)
        elif action == "status":
            cmd_status(client)
        elif action == "gpu":
            cmd_gpu(client)
        elif action == "log":
            cmd_log(client)
        elif action == "run":
            cmd = " ".join(sys.argv[2:])
            run_cmd(client, cmd)
        else:
            print(f"Unknown action: {action}")
            sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
