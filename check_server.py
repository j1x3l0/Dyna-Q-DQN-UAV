"""Check training status on server."""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('172.16.30.53', port=22, username='root', password='lingyi@2026', timeout=10)

def run(cmd, timeout=15):
    print(f'>>> {cmd}')
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    if out: print(out)
    if err: print('[stderr]', err[:300])
    print()

run('ps aux | grep run_full_benchmark | grep -v grep')
run('ps aux | grep python | grep -v grep')
run('ls -lt /root/Dyna-Q-DQN-UAV/benchmark_*.log 2>/dev/null | head -10')
run('tail -3 /root/Dyna-Q-DQN-UAV/benchmark_*.log 2>/dev/null || echo "no logs"')
run('nvidia-smi | head -15')

client.close()
