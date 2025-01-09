
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
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '1'

from torch import Tensor
class FReLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        
        
    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.relu(input) + nn.functional.relu(-input)
    
    def __repr__(self) -> str:
        return 'FReLU()'


def parse_list(arg_string):
    try:
        # Remove spaces and parse as list of integers
        return [int(item) for item in arg_string.strip('[]').split(',')]
    except:
        raise argparse.ArgumentTypeError("List must be of integers")
    
def run_tune(args):
    sub_folder = f'/sl_asym_sparse_equal_/asym_sparse_{datetime.now().strftime("%Y_%m_%d")}'
    if 'talapas' in socket.gethostname():
        path = f'/home/mtrappet/tau/BranchGatingProject/data/Rotate_LongSequence_talapas/{sub_folder}/'
    else:
        path = f'/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/{sub_folder}/'
    param_config = BASE_CONFIG.copy()
    param_config['file_path'] = path
    param_config['model_name'] = args.model_name
    param_config['n_repeat'] = args.repeat_num
    param_config['rotation_degrees'] = [0, 270, 45, 135, 225, 350, 180, 315, 60, 150, 240, 330, 90]
    param_config['n_b_1'] = args.n_branches
    param_config['epochs_per_task'] = 20
    param_config['n_eval_tasks'] = 3
    param_config['learn_gates'] = False
    param_config['sparsity'] = (args.sparsity, args.sparsity)
    param_config['soma_func'] = args.soma_func
    param_config['hidden'] = args.hidden
    gates_map = {0: False, 1: True}
    param_config['det_masks'] = gates_map[args.determ_gates]
    param_config['n_npb'] = args.n_npb
    param_config['fixed_weights'] = args.fixed_nnpb
    param_config['learning_rule'] = args.learning_rule
    if args.fixed_nnpb == 1:
        del param_config['n_npb']
    act_map = {'ReLU': nn.ReLU, 'LeakyReLU': nn.LeakyReLU, 'FReLU': FReLU}
    param_config['act_func'] = act_map[args.act_func]
    param_config['dropout'] = 0.0
    param_config['lr'] = args.lr
    
    
    run_continual_learning(param_config)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run a set of experiments with varying parameters")
    parser.add_argument('--model_name', type=str, default="Branch", help='Model name')
    parser.add_argument('--n_branches', type=int, default="1", help='Branch number index')
    parser.add_argument('--soma_func', type=str, default="sum", help='Type of soma function')
    parser.add_argument('--sparsity', type=float, default="0.0", help='Sparsity level')
    parser.add_argument('--sparsity2', type=float, default="0.0", help='Sparsity level for layer 2')
    parser.add_argument('--repeat_num', type=int, default="1", help='Number of repeats')
    parser.add_argument('--hidden', type=parse_list, default="[784, 784]", help='Hidden units and layers')
    parser.add_argument('--n_npb', type=int, default="1", help='Number of neurons per branch')
    parser.add_argument('--fixed_nnpb', type=int, default="1", help='Fixed weights')
    parser.add_argument('--act_func', type=str, default="ReLU", help='Activation function')
    parser.add_argument('--determ_gates', type=int, default="0", help='Deterministic gates')
    parser.add_argument('--learning_rule', type=str, default="sl", help='sl or rl')
    parser.add_argument('--lr', type=float, default="0.0001", help='Learning rate')
    # Add other parameters as needed

    args = parser.parse_args()
    run_tune(args)

