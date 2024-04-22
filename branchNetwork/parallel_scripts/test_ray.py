import torch
import numpy as np
import ray
import os
import socket


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