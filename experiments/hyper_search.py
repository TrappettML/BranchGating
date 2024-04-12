from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.dataloader import load_rotated_flattened_mnist_data
from tests.GatingBranchRotatingMNIST import train_epoch, evaluate_model


import torch
import torch.nn as nn
import ray
from ray import tune, train
import pandas as pd

from typing import Union
from ipdb import set_trace
import time


MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
MODEL_CONFIGS = {'n_in': 784, 
                     'n_out': 10, 
                     'n_contexts': 1, 
                     'device': 'cpu', 
                     'n_npb': [56, 200], 
                     'n_branches': [14, 10], 
                     'sparsity': 0.8,
                     'dropout': 0.5,}
    
def train_and_evaluate_model(configs: dict[str, Union[str, int]]) -> float:
    model = configs['model_class'](configs['model_configs'])
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = load_rotated_flattened_mnist_data(batch_size= configs['batch_size'],
                                                                  rotation_in_degrees=configs['rotation_in_degrees'])
    for epoch in range(configs['n_epochs']):
        train_epoch(model, train_loader, configs['rotation_in_degrees'], optimizer, criterion, device='cpu')
        accuracy, _ = evaluate_model(model, configs['rotation_in_degrees'], test_loader, criterion)
        train.report(dict(mean_accuracy=accuracy))
    return accuracy

def run_tune():
    if not ray.is_initialized():
        ray.init(num_cpus=20)
    tuner = tune.Tuner(
        tune.with_resources(train_and_evaluate_model, {"cpu": 2}),
        param_space={
            "model_class": tune.grid_search(MODEL_CLASSES),
            "model_configs": MODEL_CONFIGS,
            "lr": tune.grid_search([0.001, ]),
            "batch_size": tune.grid_search([32,]),
            "n_epochs": 1,
            "rotation_in_degrees": 0,
        },
        tune_config=tune.TuneConfig(num_samples=1, metric="mean_accuracy", mode="max"),
    )
    results = tuner.fit()
    ray.shutdown()
    print(f'Best result: {results.get_best_result()}')
    
    return results.get_results().get_dataframe()

def process_results(results: pd.DataFrame):
    results.to_pickle('/home/users/MTrappett/mtrl/BranchGatingProject/data/hyper_search/hyper_search_results.pkl')
    print(f'Saved results to /home/users/MTrappett/mtrl/BranchGatingProject/data/hyper_search/hyper_search_results.pkl')
    
def main():
    time_start = time.time()
    results = run_tune()
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    process_results(results)
    
    
if __name__ == "__main__":
    main()