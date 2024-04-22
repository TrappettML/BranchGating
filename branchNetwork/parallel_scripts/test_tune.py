import torch
import numpy as np
import ray
from ray import tune, train
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

# Setup and run the Ray Tune experiment
analysis = tune.Tuner(
    tune.with_resources(square, {"cpu": 1}),
    config={"x": tune.grid_search([i for i in range(num_cpus_in_ray)])},  # Running as many trials as CPUs
    # verbose=1  # Increase verbosity for detailed output
)

# Print out the results of the experiments
print("Results:")
for result in analysis.results:
    print(result)