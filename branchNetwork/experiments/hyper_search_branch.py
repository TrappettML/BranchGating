
from branchNetwork.experiments.branchCLMetrics import run_continual_learning

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
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'


def run_tune():
    # MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['ExpertModel', 'MasseModel', 'SimpleModel']
    MODEL_NAMES = ['BranchModel']
    layer_1_branches = [1,2,7,14,28,49,98,196,392,784]
    layer_2_branches = [1,2,5,10,20,50,100,200,500,1000,2000]
    # layer_2_branches = [2, 10, 500, 1000]
    # layer_1_branches = [1,2]
    # layer_2_branches = [1,2]
    repeats = 10
    if not ray.is_initialized():
        if 'talapas' in socket.gethostname():
            ray.init(address='auto')
        else:
            ray.init(num_cpus=20)
    tuner = tune.Tuner(
        tune.with_resources(run_continual_learning, {"cpu": 1}),
        param_space={
            "model_name": tune.grid_search(MODEL_NAMES),
            "n_b_1": tune.grid_search(layer_1_branches), # 14, # 
            "n_b_2":  tune.grid_search(layer_2_branches), # 20, #
            "n_repeat": tune.grid_search([i for i in range(repeats)]),
            "lr": 0.0001,
            "batch_size": 32,
            "epochs_per_task": 20,
            "rotation_in_degrees": [0],
        },
        tune_config=tune.TuneConfig(num_samples=1, 
                                    metric="remembering", 
                                    mode="max"),
        run_config=train.RunConfig(name='branch_search_')
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
    print(f'_____Finsihed_____')
    
    
if __name__ == "__main__":
    main()