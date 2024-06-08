
from branchNetwork.experiments.LongTaskSequencePermuteCLMetrics import run_continual_learning

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
import copy
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

@ray.remote(num_gpus=0.01, num_cpus=1)
def ray_run_continual_learning(configs):
    local_config = copy.deepcopy(BASE_CONFIG)
    local_config.update(configs)
    return run_continual_learning(local_config)
    
def run_tune():
    # MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['MasseModel']
    MODEL_NAMES = ['BranchModel']
    layer_1_branches = [1,2,7] # [1,2,7,14,28,49,98,196,392,784]
    # layer_2_branches = [2, 10, 500, 1000]
    # layer_1_branches = [1,2]
    # layer_2_branches = [1,2]
    repeats = 1
    device = torch.device('cuda')
    print(torch.cuda.is_available())
    if not ray.is_initialized():
        if 'talapas' in socket.gethostname():
            ray.init(address='auto')
        elif 'voltar' in socket.gethostname():
            ray.init(num_cpus=10, num_gpus=3)
        else:
            ray.init(num_cpus=10, num_gpus=1)
    repeats = [i for i in range(repeats)]
    permute_seeds = [0, 180, 90, 270, 45, 135, 225, 315, 60, 150, 240, 330]
    epochs_per_task = 2
    learn_gates = False
    
    sparsities = [0.8] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # param_config['gate_func'] = tune.grid_search(['sum', 'max', 'softmax', 'softmax_sum'])
    results = ray.get([ray_run_continual_learning.remote({'model_name': model_name,
                                                          'n_repeat': repeat,
                                                          'permute_seeds': permute_seeds,
                                                          'n_b_1': layer_1_branch,
                                                          'epochs_per_task': epochs_per_task,
                                                          'learn_gates': learn_gates,
                                                          'device': device,
                                                          'sparsity': sprasity,}) 
                       for model_name in MODEL_NAMES 
                       for repeat in repeats 
                       for layer_1_branch in layer_1_branches
                       for sprasity in sparsities])
    ray.shutdown()
    


def process_results(results: pd.DataFrame, file_name):
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/hyper_search/LongSequence/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/LongSequence/'
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_pickle(f'{path}/{file_name}.pkl')
    print(f'Saved results to {path}/{file_name}.pkl')
    
def main():
    time_start = time.time()
    results = run_tune()
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    process_results(results, 'gpu_test_results')
    print(f'_____Finsihed_____')
    
    
if __name__ == "__main__":
    main()