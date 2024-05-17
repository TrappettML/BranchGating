
import torch
from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.simpleMLP import SimpleMLP
from branchNetwork.BranchLayer import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from branchNetwork.dataloader import load_permuted_flattened_mnist_data
from branchNetwork.utils.timing import timing_decorator

from torch.utils.data import DataLoader
from torch import nn
from ipdb import set_trace
import ray
from ray import tune, train
from collections import OrderedDict
from typing import Callable, Union
from pickle import dump
import os

# Function to train the model for one epoch
def train_epoch(model, data_loader, task, optimizer, criterion, device='cpu'):
    model.train()
    # print(f'begining train epoch')
    total_loss = 0
    for i, (images, labels) in enumerate(data_loader):
        if i > 2:
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
            if i>4:
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

@ray.remote
def single_ray_eval(model, images, labels, task, criterion, device='cpu'):
    model.eval()
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, task)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
    return correct, loss.item(), labels.size(0)


def parallel_eval_data(model, task, data_loader, criterion, device='cpu'):
    model.eval()
    # print(f'beinging evaluation')
    correct, total = 0, 0
    total_loss = 0
    with torch.no_grad():
        results = ray.get([single_ray_eval.remote(model, images, labels, task, criterion, device=device) for i, (images, labels) in enumerate(data_loader)]) # if i < 3
        for correct_batch, loss_batch, total_batch in results:
            total += total_batch
            correct += correct_batch
            total_loss += loss_batch
        accuracy = 100 * correct / total
    # print(f'accuracy: {accuracy}; correct: {correct}; total: {total}')
    return accuracy, total_loss / len(data_loader)

@ray.remote
def new_ray_evaluate_model(model, task, test_loader, criterion, device='cpu'):
    return {task: parallel_eval_data(model, task, test_loader, criterion, device=device)}

# @ray.remote
def ray_evaluate_model(model, task, test_loader, criterion, device='cpu', tune_b=False):
    results = {task: evaluate_model(model, task, test_loader, criterion, device=device)}
    if tune_b:
        train.report({f'{task}_accurracy': results[task][0]})
    return results
     
def parallel_evaluate_model(model: nn.Module, test_loaders: dict[int, DataLoader], criterion: Callable, device='cpu'):
    results = ray.get([new_ray_evaluate_model.remote(model, task, test_loader, criterion, device=device) for i, (task, test_loader) in enumerate(test_loaders.items())]) # if i < 3
    # results = [ray_evaluate_model(model, task, test_loader, criterion, device=device) for task, test_loader in test_loaders.items()]
    return results # list of dictionaries

def merge_results_for_model(model_aggregated_results: OrderedDict, train_losses: float, test_results:list[dict[str, tuple]], train_task_name: int):
    model_aggregated_results['Training_Loss'].append(train_losses)
    for test_result in test_results:
        angle = [k for k in test_result.keys()][0] # will only be one key
        loss = test_result[angle][1]
        accuracy = test_result[angle][0]
        model_aggregated_results[f'd{angle}_loss'].append(loss)
        model_aggregated_results[f'd{angle}_Accuracy'].append(accuracy)

def single_task(model:nn.Module,
                optimizer: Callable, 
                train_loader: DataLoader, 
                train_task: Union[str, int],
                test_loaders: dict[int, DataLoader], 
                criterion: Callable, 
                epochs: int,
                model_aggregated_results: OrderedDict['str', Union[str, list[float]]], 
                device:str = 'cpu',
                tune_b: bool = False):
    model.train()
    print(f'begining training for task {train_task} on model: {model_aggregated_results["model_name"]}')
    for epoch in range(epochs):
        train_losses = train_epoch(model, train_loader, train_task, optimizer, criterion, device=device)
        if tune_b:
            train.report({'loss': train_losses})
            test_results = [ray_evaluate_model(model, task, test_loader, criterion, device=device, tune_b=tune_b) for task, test_loader in test_loaders.items()]
        else:
            test_results = parallel_evaluate_model(model, test_loaders, criterion, device=device)
        merge_results_for_model(model_aggregated_results, train_losses, test_results, train_task)
    return model_aggregated_results, train_losses

def setup_model(model_name: str, 
                model_configs: dict[str, Union[int, list[int], dict[str, int], str]], 
                model_dict: Union[None, dict[str, Callable]]):
    
    assert model_name in model_dict.keys(), f'{model_name} not in model_dict'
    model = model_dict[model_name](model_configs)   
    optim = torch.optim.Adam(model.parameters(), lr=model_configs['lr'])
    criterion = nn.CrossEntropyLoss()
    return model, optim, criterion

def setup_loaders(permute_seeds: list[int], batch_size: int):
    train_loaders = dict()
    test_loaders = dict()
    for seed in permute_seeds:
        test_loader, train_loader = load_permuted_flattened_mnist_data(batch_size, seed)
        train_loaders[seed] = train_loader
        test_loaders[seed] = test_loader
    return train_loaders, test_loaders

def make_data_container(permute_seeds: list[int], model_name: str):
    keys = ['model_name', 'Training_Loss']
    for deg in permute_seeds:
        keys.extend([f'd{deg}_loss', f'd{deg}_Accuracy'])

    values = [model_name, []]
    for _ in permute_seeds:
        values.extend([[], []])  
        
    model_data = OrderedDict(zip(keys, values))
    return model_data

@timing_decorator
@ray.remote
def train_model(model_name: str,
                train_config: dict[str, Union[int, list[int]]],
                model_dict: dict[str, Callable] = None,
                model_configs: dict[str, Union[int, float, list[int]]] = None,
                device: str = 'cpu'):

    model, optimizer, criterion = setup_model(model_name, model_configs=model_configs, model_dict=model_dict)
    train_loaders, test_loaders = setup_loaders(train_config['permute_seeds'], train_config['batch_size'])
    
    model_data = make_data_container(train_config["permute_seeds"], model_name)

    for task_name, task_loader in train_loaders.items():
        single_task(model, optimizer, task_loader, task_name, test_loaders, criterion, train_config['epochs_per_train'], 
                    model_data, device=device)
    return {'model_name': model_name, 'data': model_data}

    
def save_results(results, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    with open(f'{results_path}/3task_permute_results.pkl', 'wb') as f:
        dump(results, f)
        
@timing_decorator
def run_continual_learning():
    MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
    MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    MODEL_DICT = {name: model for name, model in zip(MODEL_NAMES, MODEL_CLASSES)}
    TRAIN_CONFIGS = {'batch_size': 32,
                    'epochs_per_train': 2,
                    'permute_seeds': [None, 21, 42],}
    
    MODEL_CONFIGS = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': len(TRAIN_CONFIGS['permute_seeds']), 
                    'device': 'cpu', 
                    'n_npb': [56, 56], 
                    'n_branches': [14, 14], 
                    'sparsity': 0.8,
                    'dropout': 0.5,
                    'hidden_layers': [784, 784],
                    'lr': 0.0001,
                    }
    
    if not ray.is_initialized():
        ray.init(num_cpus=70)
    results = ray.get([train_model.remote(model_name, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS) for model_name in MODEL_NAMES])
    ray.shutdown()
    # results = [train_model(model_name, permute_seeds, epochs_per_train)
    # for model_name in model_names]
    save_results(results, results_path='/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/results')
    print('Results saved')



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    run_continual_learning()