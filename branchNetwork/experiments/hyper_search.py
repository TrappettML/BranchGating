from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
# from branchNetwork.dataloader import load_rotated_flattened_mnist_data
from branchNetwork.dataloader import load_permuted_flattened_mnist_data

import torch
import torch.nn as nn
import ray
from ray import tune, train
import pandas as pd

from typing import Union
from ipdb import set_trace
import time
import os
import socket

# Function to train the model for one epoch
def train_epoch(model, data_loader, task, optimizer, criterion, device='cpu'):
    model.train()
    # print(f'begining train epoch')
    total_loss = 0
    for i, (images, labels) in enumerate(data_loader):
        if i > 3:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, task)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Function to evaluate the model
def evaluate_model(model, task, data_loader, criterion, device='cpu'):
    model.eval()
    # print(f'beinging evaluation')
    correct, total = 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i>3:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, task)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    accuracy = 100 * correct / total
    return accuracy, total_loss / len(data_loader)


MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
MODEL_DICT = {name: model for name, model in zip(MODEL_NAMES, MODEL_CLASSES)}
MODEL_CONFIGS = {'n_in': 784, 
                     'n_out': 10, 
                     'n_contexts': 1, 
                     'device': 'cpu', 
                     'n_npb': [56, 56], 
                     'n_branches': [14, 14], 
                     'sparsity': 0.8,
                     'dropout': 0.5,}
    
def train_and_evaluate_model(configs: dict[str, Union[str, int]]) -> float:
    model = MODEL_DICT[configs['model_name']](configs['model_configs'])
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = load_permuted_flattened_mnist_data(batch_size= configs['batch_size'],
                                                                  permute_seed=configs['permute_seed'])
    for epoch in range(configs['n_epochs']):
        train_epoch(model, train_loader, configs['permute_seed'], optimizer, criterion, device='cpu')
        accuracy, _ = evaluate_model(model, configs['permute_seed'], test_loader, criterion)
        train.report({'mean_accuracy':accuracy})
    # return accuracy

def run_tune():
    if not ray.is_initialized():
        ray.init(num_cpus=70)
    tuner = tune.Tuner(
        tune.with_resources(train_and_evaluate_model, {"cpu": 1}),
        param_space={
            "model_name": tune.grid_search(MODEL_NAMES),
            "model_configs": MODEL_CONFIGS,
            "lr": tune.grid_search([0.0001, 0.001, 0.01, 0.1 ]),
            "batch_size": tune.grid_search([32, 64, 128, 512]),
            "n_epochs": 2,
            "permute_seed": 0,
        },
        tune_config=tune.TuneConfig(num_samples=1, 
                                    metric="mean_accuracy", 
                                    mode="max"),
        run_config=train.RunConfig(name='permuted_mnist_hyper_search_lr_batch_size')
    )
    results = tuner.fit()
    ray.shutdown()
    print(f'Best result: {results.get_best_result()}')
    
    return results.get_dataframe()

def process_results(results: pd.DataFrame):
    if 'talapas' in socket.gethostname():
        path = '/home/mtrappet/branchNetwork/data/hyper_search/'
    else:
        path = '/home/users/MTrappett/mtrl/BranchGatingProject/data/hyper_search/'
    if not os.path.exists(path):
        os.makedirs('/home/mtrappet/BranchGating/branchNetwork/data/results/hyper_search/')
    results.to_pickle(f'{path}/lr_bs_hyper_search_results.pkl')
    print(f'Saved results to {path}/lr_bs_hyper_search_results.pkl')
    
def main():
    time_start = time.time()
    results = run_tune()
    elapsed_time = time.time() - time_start
    print(f'Elapsed time: {elapsed_time} seconds')
    process_results(results)
    
    
if __name__ == "__main__":
    main()