"""Deploy project to server and run benchmark training.

Credentials are read from environment variables:
  UAV_HOST, UAV_PORT, UAV_USER, UAV_PASSWORD

Uses paramiko (pure Python SSH) — works on Windows/Linux/Mac.
"""
import os
import sys
import time
from datetime import datetime

import paramiko

HOST = os.environ.get('UAV_HOST', '')
PORT = int(os.environ.get('UAV_PORT', '22'))
USER = os.environ.get('UAV_USER', 'root')
PASSWORD = os.environ.get('UAV_PASSWORD', '')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = '/root/uav-drl'


def get_client():
    """Create a new SSH client connected to the server."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   look_for_keys=False, allow_agent=False, timeout=30)
    return client


def run_ssh(cmd, timeout=120):
    """Run command on remote server and return (exit_code, stdout, stderr)."""
    client = get_client()
    try:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        exit_code = stdout.channel.recv_exit_status()
        if out:
            print(out.strip())
        if err:
            print(err.strip(), file=sys.stderr)
        return exit_code, out, err
    finally:
        client.close()


def run_scp(local_path, remote_path, timeout=600):
    """Upload file or directory to remote server."""
    import stat as _stat
    client = get_client()
    try:
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(timeout)

        if os.path.isfile(local_path):
            sftp.put(local_path, remote_path)
            print(f"  Uploaded file: {os.path.basename(local_path)} -> {remote_path}")

        elif os.path.isdir(local_path):
            # Ensure remote directory exists
            try:
                sftp.stat(remote_path)
            except FileNotFoundError:
                run_ssh(f"mkdir -p {remote_path}")

            for root, dirs, files in os.walk(local_path):
                rel_root = os.path.relpath(root, local_path)
                remote_root = os.path.join(remote_path, rel_root).replace('\\', '/') if rel_root != '.' else remote_path

                # Create remote subdirectories
                for d in dirs:
                    remote_subdir = os.path.join(remote_root, d).replace('\\', '/')
                    try:
                        sftp.mkdir(remote_subdir)
                    except IOError:
                        pass  # Directory may already exist

                # Upload files
                for f in files:
                    local_file = os.path.join(root, f)
                    remote_file = os.path.join(remote_root, f).replace('\\', '/')
                    sftp.put(local_file, remote_file)

            print(f"  Uploaded directory: {local_path} -> {remote_path}")

        sftp.close()
    finally:
        client.close()


def download_results(remote_pattern, local_dir, timeout=600):
    """Download files from remote server."""
    client = get_client()
    try:
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(timeout)
        os.makedirs(local_dir, exist_ok=True)

        # List remote files matching pattern
        remote_dir = os.path.dirname(remote_pattern)
        if '*' in remote_pattern:
            base = os.path.basename(remote_pattern).replace('*', '')
            ext = os.path.splitext(base)[1]
            try:
                files = sftp.listdir(remote_dir)
                for f in files:
                    if f.endswith(ext) or base in f:
                        remote_file = os.path.join(remote_dir, f).replace('\\', '/')
                        local_file = os.path.join(local_dir, f)
                        try:
                            sftp.get(remote_file, local_file)
                            print(f"  Downloaded: {f}")
                        except IOError as e:
                            print(f"  Skip {f}: {e}")
            except IOError:
                print(f"  Cannot list {remote_dir} (directory may be empty)")
        else:
            # Single file
            fname = os.path.basename(remote_pattern)
            sftp.get(remote_pattern, os.path.join(local_dir, fname))
            print(f"  Downloaded: {fname}")

        sftp.close()
    finally:
        client.close()


def main():
    # Validate credentials
    if not HOST:
        print("ERROR: UAV_HOST environment variable not set.")
        print("Set credentials: export UAV_HOST=... UAV_PORT=... UAV_USER=... UAV_PASSWORD=...")
        sys.exit(1)

    algo_list = os.environ.get('UAV_ALGOS', 'maddpg,nodyna,dyna')
    seed_list = os.environ.get('UAV_SEEDS', '42,123,2026')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"{'='*70}")
    print(f"UAV DRL Server Deployment")
    print(f"{'='*70}")
    print(f"Target: {USER}@{HOST}:{PORT}")
    print(f"Remote dir: {REMOTE_DIR}")
    print(f"Algorithms: {algo_list}")
    print(f"Seeds: {seed_list}")
    print(f"Timestamp: {timestamp}")
    print()

    # --- Step 1: Check connectivity and server specs ---
    print("=== Step 1: Checking server connectivity ===")
    exit_code, out, _ = run_ssh("echo 'connected' && uname -a && lscpu 2>/dev/null | grep 'Model name' || echo 'CPU info N/A' && free -h 2>/dev/null | head -2 || echo 'RAM info N/A'")
    if exit_code != 0:
        print(f"SSH connection failed (exit={exit_code})")
        sys.exit(1)

    # Check GPU
    print("\n  Checking GPU...")
    run_ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4 || echo '  No GPU detected'")

    # --- Step 2: Prepare remote directory ---
    print("\n=== Step 2: Preparing remote directory ===")
    run_ssh(f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}/src {REMOTE_DIR}/scripts {REMOTE_DIR}/results {REMOTE_DIR}/checkpoints {REMOTE_DIR}/logs")

    # --- Step 3: Upload project files ---
    print("\n=== Step 3: Uploading project files ===")
    # Upload individual source files (not the whole src directory with __pycache__)
    for f in ['system_model.py', 'maddpg_agent.py', 'hierarchical_agent.py', 'iddpg_agent.py']:
        run_scp(os.path.join(PROJECT_DIR, 'src', f), f'{REMOTE_DIR}/src/{f}')

    for f in ['training_utils.py', 'run_full_benchmark.py']:
        run_scp(os.path.join(PROJECT_DIR, 'scripts', f), f'{REMOTE_DIR}/scripts/{f}')

    run_scp(os.path.join(PROJECT_DIR, 'requirements.txt'), f'{REMOTE_DIR}/requirements.txt')

    # --- Step 4: Install dependencies ---
    print("\n=== Step 4: Installing dependencies ===")
    run_ssh(f"cd {REMOTE_DIR} && pip install numpy matplotlib torch -q 2>&1 | tail -3", timeout=600)

    # --- Step 5: Verify installation ---
    print("\n=== Step 5: Verifying installation ===")
    run_ssh(f"cd {REMOTE_DIR} && python -c 'import torch; print(f\"torch {torch.__version__}, cuda={torch.cuda.is_available()}\"); import numpy; print(f\"numpy {numpy.__version__}\")'")

    # --- Step 6: Quick smoke test ---
    print("\n=== Step 6: Quick smoke test (3 eps each, 1 seed) ===")
    run_ssh(
        f"cd {REMOTE_DIR} && python -c \""
        f"import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); "
        f"from run_full_benchmark import run_single_experiment, ALGO_CONFIGS; "
        f"for a in ['maddpg', 'nodyna', 'dyna']: "
        f"  ALGO_CONFIGS[a]['max_episodes'] = 3; "
        f"  ALGO_CONFIGS[a]['checkpoint_every'] = 10; "
        f"  r = run_single_experiment(a, seed=42); "
        f"  print(f'  {a}: {r.episodes_completed} eps, stopped={r.stopped_early}, {r.duration:.1f}s')\"",
        timeout=300
    )

    # --- Step 7: Run full benchmark ---
    print(f"\n=== Step 7: Running full benchmark training ===")
    estimated_hours = {
        'iddpg,maddpg,nodyna,dyna': '40-50 GPU hours total (3 seeds each)',
        'maddpg,nodyna,dyna': '30-40 GPU hours total (3 seeds each)',
        'dyna': '15-20 GPU hours (3 seeds)',
    }
    algo_key_for_est = algo_list if algo_list in estimated_hours else 'maddpg,nodyna,dyna'
    print(f"  Estimated time: {estimated_hours.get(algo_key_for_est, 'varies')}")
    print(f"  Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start = time.time()
    exit_code, out, err = run_ssh(
        f"cd {REMOTE_DIR} && python scripts/run_full_benchmark.py "
        f"--algos {algo_list} --seeds {seed_list} --case 1",
        timeout=172800  # 48 hours
    )

    elapsed = time.time() - start
    print(f"\n  Training finished! Elapsed: {elapsed/3600:.1f} hours")

    # --- Step 8: Download results ---
    print("\n=== Step 8: Downloading results ===")
    local_results = os.path.join(PROJECT_DIR, 'results')
    local_ckpts = os.path.join(PROJECT_DIR, 'checkpoints')
    os.makedirs(local_results, exist_ok=True)
    os.makedirs(local_ckpts, exist_ok=True)

    download_results(f'{REMOTE_DIR}/results/*', local_results)
    download_results(f'{REMOTE_DIR}/checkpoints/*', local_ckpts)

    # Also download the latest log
    print("\n=== Downloading logs ===")
    local_logs = os.path.join(PROJECT_DIR, 'logs')
    os.makedirs(local_logs, exist_ok=True)
    download_results(f'{REMOTE_DIR}/logs/*', local_logs)

    print(f"\n{'='*70}")
    print(f"Deployment complete!")
    print(f"Results: {local_results}")
    print(f"Checkpoints: {local_ckpts}")
    print(f"Logs: {local_logs}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
