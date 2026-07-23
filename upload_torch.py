"""SFTP upload to matpool - not affected by MOTD."""
import paramiko
import os
import sys

host = 'hz-4.matpool.com'
port = 29989
user = 'root'
password = 'ESr0aj=)zusB{DXN'

local_file = r'C:\Users\Lenovo\Downloads\torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl'

if not os.path.exists(local_file):
    print(f"File not found: {local_file}")
    sys.exit(1)

file_size = os.path.getsize(local_file) / (1024**3)
print(f"Uploading {local_file} ({file_size:.1f} GB)...")

transport = paramiko.Transport((host, port))
transport.connect(username=user, password=password)

sftp = paramiko.SFTPClient.from_transport(transport)
sftp.put(local_file, '/tmp/torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl',
         callback=lambda x, y: print(f'\r  {x/1024**2:.0f}/{file_size*1024:.0f} MB', end=''))
sftp.close()
transport.close()
print("\nDone!")
