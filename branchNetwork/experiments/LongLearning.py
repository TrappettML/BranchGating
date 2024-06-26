
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
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'


def run_tune():
    # MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['MasseModel']
    MODEL_NAMES = ['BranchModel']
    layer_1_branches =  [14] # [1, 2, 7, 14, 28, 49, 98, 196, 382, 784] # [14] # [49,98,196,392,784] # [1,2,7,14,] 
    sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1.0]
    # layer_2_branches = [2, 10, 500, 1000]
    # layer_1_branches = [1,2]
    # layer_2_branches = [1,2]
    repeats = 1
    repeats_list = [i for i in range(repeats)]
    
    # Helper function to split the list into chunks
    def chunk_list(lst, n):
        """Yield successive n-sized chunks from lst."""
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    # Define chunk size
    chunk_size = 7
    
    if not ray.is_initialized():
        if 'talapas' in socket.gethostname():
            ray.init(address='auto')
        else:
            ray.init(num_cpus=140)
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/Rotate_LongSequence_talapas/soma_func_softmax_temp/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/Rotate_LongSequence/soma_func_softmax_temp/'
    param_config = BASE_CONFIG
    param_config['file_path'] = path
    param_config['model_name'] = tune.grid_search(MODEL_NAMES)
    param_config['n_repeat'] = tune.grid_search(repeats_list) # tune.grid_search([0, 2, 3, 4])
    param_config['rotation_degrees'] = [0, 180, 90, 270, 45, 135, 225, 315, 60, 150, 240, 330]
    param_config['n_b_1'] = tune.grid_search(layer_1_branches)
    param_config['epochs_per_task'] = 20
    param_config['n_eval_tasks'] = 3
    param_config['learn_gates'] = tune.grid_search([False])
    # param_config['sparsity'] = tune.grid_search([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) #tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, ]) # 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    param_config['sparsity'] = tune.grid_search(sparsity)
    param_config['soma_func'] = tune.grid_search(['softmax']) # tune.grid_search(['max']) # , 'softmax_sum' # lets do softmax_sum after with different temps # 'sum', 'max', 'softmax', 'softmax_sum'
    param_config['temp'] = tune.grid_search([0.1, 1, 10])
    
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
        run_config=train.RunConfig(name='Soma_func_max')
    )
    results = tuner.fit()
    ray.shutdown()
    print(f'Best result: {results.get_best_result()}')
    return results.get_dataframe()

def process_results(results: pd.DataFrame, file_name):
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/hyper_search/Rotate_LongSequence4/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/hyper_search/Rotate_LongSequence4/'
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_pickle(f'{path}/{file_name}.pkl')
    print(f'Saved results to {path}/{file_name}.pkl')
    
def main(chunk_num: int=0):
    time_start = time.time()
    results = run_tune(chunk_num)
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    process_results(results, 'sparsity_LearnGates_results')
    print(f'_____Finsihed_____')
    
    
if __name__ == "__main__":
    chunk_num = sys.argv[1]
    main(int(chunk_num))