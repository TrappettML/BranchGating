import torch
import numpy as np
import ray
import os
import socket
import subprocess

def run_nvidia_smi():
    # Command to run nvidia-smi
    command = ["nvidia-smi"]

    # Open a subprocess and run the command
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        output, errors = proc.communicate()

    # Print the outputs
    if proc.returncode == 0:
        print("nvidia-smi output:")
        print(output)
    else:
        print("Error running nvidia-smi:")
        print(errors)

# Call the function
run_nvidia_smi()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(f'Number of GPUs from torch: {torch.cuda.device_count()}')

ray.init(address='auto')
print(f'Ray gpus: {ray.get_gpu_ids()}')
@ray.remote
def square(x):
    return x * x, os.getpid(), socket.gethostname()

print(f'Ray cluster resources: {ray.cluster_resources()}')
num_cpus_in_ray = int(ray.cluster_resources()['CPU'])

# Launch four parallel square tasks.
futures = [square.remote(i) for i in range(num_cpus_in_ray)]

squares, pids, hostnames = [list(l) for l in zip(*ray.get(futures))]

print(f'squares: {squares}\npids: {set(pids)}, len: {len(set(pids))}\nhostnames: {set(hostnames)}, len: {len(set(hostnames))}')