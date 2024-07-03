
from branchNetwork.experiments.LongTaskSequenceRotate import run_continual_learning

import torch
import torch.nn as nn

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


def run_tune(args):
    # MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['MasseModel']
    MODEL_NAMES = ['BranchModel']
    branches =  [1, 2, 7, 14, 28, 49, 98, 196, 382, 784] # [14] # [49,98,196,392,784] # [1,2,7,14,] 
    sparsity = [0] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1.0]
    # layer_2_branches = [2, 10, 500, 1000]
    # layer_1_branches = [1,2]
    # layer_2_branches = [1,2]
    repeats = 3
    repeats_list = [i for i in range(repeats)]
    soma_funcs = ['softmax_0.1', 'softmax_1.0', 'softmax_2.0', 
                'softmaxsum_0.1', 'softmaxsum_0.5', 'softmaxsum_1.0', 'softmaxsum_2.0',
                'max', 'sum', 'median']

    
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/Rotate_LongSequence_talapas/soma_func_branch_search/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/Rotate_LongSequence/soma_func_branch_search/'
    param_config = BASE_CONFIG
    param_config['file_path'] = path
    param_config['model_name'] = args.model_name
    param_config['n_repeat'] = args.repeat_num
    param_config['rotation_degrees'] = [0, 180, 90] # , 270, 45, 135, 225, 315, 60, 150, 240, 330]
    param_config['n_b_1'] = args.branch_num
    param_config['epochs_per_task'] = 20
    param_config['n_eval_tasks'] = 3
    param_config['learn_gates'] = False
    param_config['sparsity'] = sparsity

    param_config['soma_func'] = args.soma_func
    
    run_continual_learning(param_config)
    # need to allocate cpus for sub processes see: 
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.execution.placement_groups.PlacementGroupFactory.html#ray.tune.execution.placement_groups.PlacementGroupFactory

    

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a set of experiments with varying parameters")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--branch_num', type=int, required=True, help='Branch number index')
    parser.add_argument('--soma_func', type=str, required=True, help='Type of soma function')
    parser.add_argument('--sparsity', type=float, required=True, help='Sparsity level')
    parser.add_argument('--repeat_num', type=int, required=True, help='Number of repeats')
    # Add other parameters as needed

    args = parser.parse_args()
    run_tune(args)

