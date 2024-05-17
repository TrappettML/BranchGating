
from branchNetwork.tests.GatingBranchPermuteMNIST import setup_model, setup_loaders, make_data_container, single_task
from branchNetwork.dataloader import load_permuted_flattened_mnist_data
from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel

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
from typing import Callable
from datetime import datetime

os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

dt = datetime.now().strftime('%m_%d_%H_%M_%S')


def train_model(config: dict):
    model_name = config['model_name']
    model_configs = {'n_in': config['n_in'], 'n_out': config['n_out'], 'n_contexts': config['n_contexts'], 
                     'n_npb': config['n_npb'], 'n_branches': config['n_branches'], 'sparsity': config['sparsity'],
                     'dropout': config['dropout'], 'hidden_layers': config['hidden_layers'], 'lr': config['lr'], 'device': config['device']}
    train_config = {'lr': config['lr'], 'batch_size': config['batch_size'], 'epochs_per_train': config['epochs_per_train'],
                    'permute_seeds': config['permute_seeds']}
    MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
    MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    
    MODEL_DICT = {name: model for name, model in zip(MODEL_NAMES, MODEL_CLASSES)}
    model, optimizer, criterion = setup_model(model_name, model_configs=model_configs, model_dict=MODEL_DICT)
    train_loaders, test_loaders = setup_loaders(train_config['permute_seeds'], train_config['batch_size'])
    
    model_data = make_data_container(train_config["permute_seeds"], model_name)
    tot_ave_loss = 0
    for task_name, task_loader in train_loaders.items():
        _, l = single_task(model, optimizer, task_loader, task_name, test_loaders, criterion, train_config['epochs_per_train'], 
                    model_data, tune_b=True)
        tot_ave_loss += l
    return {'model_name': model_name, 'data': model_data}


def run_tune():
    MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['ExpertModel', 'MasseModel', 'SimpleModel']
    # MODEL_NAMES = ['BranchModel']
    # branch_options = [1,2,7,14,28,49,98,196,392,784]
    seeds = [None, 21, 42]
    repeats = 125
    param_space = {'model_name':tune.grid_search(MODEL_NAMES), 
                   'repeat':tune.grid_search([i for i in range(repeats)]),
                   'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': len(seeds), 
                    'device': 'cpu', 
                    'n_npb': [56, 56], 
                    'n_branches': [14, 14], 
                    'sparsity': 0.8,
                    'dropout': 0.5,
                    'hidden_layers': [784, 784],
                    'lr': 0.0001,
                    'batch_size': 32,
                    'epochs_per_train': 40,
                    'permute_seeds': seeds,
                    'device': 'cpu'
                    }
    if not ray.is_initialized():
        if 'talapas' in socket.gethostname():
            ray.init(address='auto')
        else:
            ray.init(num_cpus=20)
    tuner = tune.Tuner(
        tune.with_resources(train_model, {"cpu": 1}),
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=train.RunConfig(name=f'3_task_permute_{dt}')
    )
    results = tuner.fit()
    ray.shutdown()
    # print(f'Best result: {results.get_best_result()}')
    return results.get_dataframe()

def process_results(results: pd.DataFrame, file_name):
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/BranchGatingProject/branchNetwork/data/3_task_permutation/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/3_task_permutation/'
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_pickle(f'{path}/{file_name}.pkl')
    print(f'Saved results to {path}/{file_name}.pkl')
    
def main():
    time_start = time.time()
    results = run_tune()
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    process_results(results, f'3_task_permute_results_{dt}')
    print(f'_____Finsihed_____')
    
    
if __name__ == "__main__":
    main()