
from branchNetwork.experiments.LongTaskSequenceRotate import run_continual_learning

import torch
import torch.nn as nn
import ray
from ray import tune, train
import pandas as pd
from branchNetwork.configs import BASE_CONFIG

from typing import Union
from ipdb import set_trace
import time
import socket
import os
import sys
import itertools
import argparse

os.environ['RAY_AIR_NEW_OUTPUT'] = '0'


def run_tune():
    # MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['MasseModel']
    MODEL_NAMES = ['BranchModel']
    branches =  [1, 2, 7, 14, 28, 49, 98, 196, 382, 784] # [14] # [49,98,196,392,784] # [1,2,7,14,] 
    sparsity = [0] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1.0]
    # layer_2_branches = [2, 10, 500, 1000]
    # layer_1_branches = [1,2]
    # layer_2_branches = [1,2]
    repeats = 1
    repeats_list = [i for i in range(repeats)]
    soma_funcs = ['softmax_0.1', 'softmax_1.0', 'softmax_2.0', 
                'softmaxsum_0.1', 'softmaxsum_0.5', 'softmaxsum_1.0', 'softmaxsum_2.0',
                'max', 'sum', 'median']

    
    if not ray.is_initialized():
        if 'talapas' in socket.gethostname():
            ray.init(address='auto')
        else:
            ray.init(num_cpus=140)
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/Rotate_LongSequence_talapas/soma_func_branch_search2/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/Rotate_LongSequence/soma_func_branch_search2/'
    param_config = BASE_CONFIG
    param_config['file_path'] = path
    param_config['model_name'] = tune.grid_search(MODEL_NAMES)
    param_config['n_repeat'] = tune.grid_search(repeats_list) # tune.grid_search([0, 2, 3, 4])
    param_config['rotation_degrees'] = [0, 180, 90, 270, 45, 135, 225, 315, 60, 150, 240, 330]
    param_config['n_b_1'] = tune.grid_search(branches)
    param_config['epochs_per_task'] = 20
    param_config['n_eval_tasks'] = 3
    param_config['learn_gates'] = tune.grid_search([False])
    param_config['sparsity'] = tune.grid_search(sparsity)

    param_config['soma_func'] = tune.grid_search(soma_funcs) 
 
    # need to allocate cpus for sub processes see: 
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.execution.placement_groups.PlacementGroupFactory.html#ray.tune.execution.placement_groups.PlacementGroupFactory

    tuner = tune.Tuner(
        tune.with_resources(run_continual_learning, 
                            resources=tune.PlacementGroupFactory(
                                [{"CPU": 1}] + [{"CPU": 1}]*param_config['n_eval_tasks'] 
                                )
                            ),
        param_space=param_config,
        tune_config=tune.TuneConfig(num_samples=1, 
                                    metric="forward_transfer", 
                                    mode="max"),
        run_config=train.RunConfig(name=f'SomaFuncSearch_BranchSearch')
    )
    results = tuner.fit()
    ray.shutdown()
    print(f'Best result: {results.get_best_result()}')
    return results.get_dataframe()
    
def main(chunk_num: int=0):
    time_start = time.time()
    run_tune(chunk_num)
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    # process_results(results, 'sparsity_LearnGates_results')
    print(f'_____Finsihed_____')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which branch number to run")

    # Add an argument for a single integer
    parser.add_argument('number', type=int, help='an integer input')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main()