"""Full server setup: pip + git clone + PyTorch CUDA + deps."""
import paramiko, time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('172.16.30.53', port=22, username='root', password='lingyi@2026', timeout=10)

def run(cmd, timeout=30):
    print(f'>>> {cmd}')
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    if out: print(out[-500:])
    if err: print('[stderr]', err[-300:])
    print('---')
    return out

# Step 1: Install pip
run('apt-get update -qq && apt-get install -y python3-pip git', timeout=180)

# Step 2: Clone repo
run('git clone https://github.com/j1x3l0/Dyna-Q-DQN-UAV.git /root/Dyna-Q-DQN-UAV', timeout=60)

# Step 3: Install PyTorch CUDA 11.8
run('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118', timeout=900)

# Step 4: Install deps
run('cd /root/Dyna-Q-DQN-UAV && pip3 install -r requirements.txt', timeout=60)

# Step 5: Verify
run('python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")"')

client.close()
print("Setup complete!")
