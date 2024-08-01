
from branchNetwork.experiments.FuturesLongTaskSequenceRotate import run_continual_learning

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

os.environ['OMP_NUM_THREADS'] = '2'

def run_tune(args):
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/Rotate_LongSequence_talapas/SomaFunc_x_Sparse_repeats_all_branches_newTaskOrder/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/Rotate_LongSequence/SomaFunc_x_Sparse_repeats_all_branches_newTaskOrder/'
    param_config = BASE_CONFIG.copy()
    param_config['file_path'] = path
    param_config['model_name'] = args.model_name
    param_config['n_repeat'] = args.repeat_num
    param_config['rotation_degrees'] = [0, 270, 45, 135, 225, 350, 180, 315, 60, 150, 240, 330, 90]
    param_config['n_b_1'] = args.branch_num
    param_config['epochs_per_task'] = 20
    param_config['n_eval_tasks'] = 3
    param_config['learn_gates'] = False
    param_config['sparsity'] = args.sparsity
    param_config['soma_func'] = args.soma_func
    
    run_continual_learning(param_config)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a set of experiments with varying parameters")
    parser.add_argument('--model_name', type=str, default="BranchModel", help='Model name')
    parser.add_argument('--branch_num', type=int, default="1", help='Branch number index')
    parser.add_argument('--soma_func', type=str, default="sum", help='Type of soma function')
    parser.add_argument('--sparsity', type=float, default="0.0", help='Sparsity level')
    parser.add_argument('--repeat_num', type=int, default="1", help='Number of repeats')
    # Add other parameters as needed

    args = parser.parse_args()
    run_tune(args)

