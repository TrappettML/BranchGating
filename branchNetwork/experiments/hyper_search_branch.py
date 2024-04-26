
from branchNetwork.experiments.branchCLMetrics import run_continual_learning
from branchNetwork.dataloader import load_rotated_flattened_mnist_data



import torch
import torch.nn as nn
import ray
from ray import tune, train
import pandas as pd

from typing import Union
from ipdb import set_trace
import time
import socket
import os


def run_tune():
    # layer_1_branches = [1,2,7,14,28,49,98,196,392,784]
    # layer_2_branches = [1,2,5,10,20,50,100,200,500,1000,2000]
    layer_1_branches = [1,2]
    layer_2_branches = [1,2]
    if not ray.is_initialized():
        if 'talapas' in socket.gethostname():
            ray.init(address='auto')
        else:
            ray.init(num_cpus=70)
    tuner = tune.Tuner(
        tune.with_resources(run_continual_learning, {"cpu": 2}),
        param_space={
            "n_b_1": tune.grid_search(layer_1_branches),
            "n_b_2": tune.grid_search(layer_2_branches),
            "lr": 0.001,
            "batch_size": 32,
            "epochs_per_task": 20,
            "rotation_in_degrees": [0,180],
        },
        tune_config=tune.TuneConfig(num_samples=1, 
                                    metric="remembering", 
                                    mode="max"),
    )
    results = tuner.fit()
    ray.shutdown()
    print(f'Best result: {results.get_best_result()}')
    return results.get_dataframe()

def process_results(results: pd.DataFrame, file_name):
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/hyper_search/BranchSearch/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/hyper_search/BranchSearch/'
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_pickle(f'{path}/{file_name}.pkl')
    print(f'Saved results to {path}/{file_name}.pkl')
    
def main():
    time_start = time.time()
    results = run_tune()
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    process_results(results, 'branch_search_results')
    
    
if __name__ == "__main__":
    main()