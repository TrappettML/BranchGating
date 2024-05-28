import torch
from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.dataloader import load_permuted_flattened_mnist_data
from branchNetwork.utils.timing import timing_decorator

from torch.utils.data import DataLoader
from torch import nn
from ipdb import set_trace
import ray
from ray import train
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
        # if i > 2:
        #     break
        # set_trace()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, task)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, task, data_loader, criterion, device='cpu'):
    model.eval()
    # print(f'beinging evaluation')
    correct, total = 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            # if i>2:
            #     break
            # set_trace()
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
        results = ray.get([single_ray_eval.remote(model, images, labels, task, criterion, device=device) for i, (images, labels) in enumerate(data_loader) if i < 3]) # if i < 3
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
def ray_evaluate_model(model, task, test_loader, criterion, device='cpu'):
    return {task: evaluate_model(model, task, test_loader, criterion, device=device)}
     
def parallel_evaluate_model(model: nn.Module, test_loaders: dict[int, DataLoader], criterion: Callable, device='cpu'):
    # results = ray.get([new_ray_evaluate_model.remote(model, task, test_loader, criterion, device=device) for i, (task, test_loader) in enumerate(test_loaders.items())]) # if i < 3
    results = [ray_evaluate_model(model, task, test_loader, criterion, device=device) for task, test_loader in test_loaders.items()]
    return results # list of dictionaries

def calc_remembering(A_acc_1: float, A_acc_2: float) -> float:
    """Rem = Acc_2 / Acc_1 -> this will yield a value between 0 and 1 (most likely)
       since Acc_2 is after trainging on a new task. If Greater than 1 than we have Backwards Transfer."""
    return A_acc_2/A_acc_1

def calc_forward_transfer(B_acc_0: float, B_acc_1: float) -> float:
    """FT = (B_acc_0 - B_acc_1)/ (B_acc_0 + B_acc_1)
        This will yield a value between -1 and 1. if the value is negative than negative interference.
        if positive than forward transfer of information."""
    return (B_acc_1 - B_acc_0)/ (B_acc_0 + B_acc_1)

    
def gather_task_accuracies(all_test_results: dict[str, list[float]], test_results: list[dict[int, tuple[float, float]]]) -> list[float]:
    for result in test_results:
        for task, (acc, _) in result.items():
            all_test_results.setdefault(task, []).append(acc)
    return all_test_results

@timing_decorator
def single_task(model:nn.Module,
                optimizer: Callable, 
                train_loader: DataLoader, 
                train_task: Union[str, int],
                test_loaders: dict[int, DataLoader], 
                criterion: Callable, 
                epochs: int,
                # model_aggregated_results: OrderedDict['str', Union[str, list[float]]], 
                device:str = 'cpu'):
    # model.train()
    task_accuracies = {}
    print(f'begining training model {model} on task {train_task}')
    for epoch in range(epochs):
        _ = train_epoch(model, train_loader, train_task, optimizer, criterion, device=device)
        test_results = parallel_evaluate_model(model, test_loaders, criterion, device=device)
        task_accuracies = gather_task_accuracies(task_accuracies, test_results)
    return task_accuracies

def setup_model(model_name: str, 
                model_configs: dict[str, Union[int, list[int], dict[str, int], str]], 
                model_dict: Union[None, dict[str, Callable]]):
    
    assert model_name in model_dict.keys(), f'{model_name} not in model_dict'
    model = model_dict[model_name](model_configs)   
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
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

# @timing_decorator
# @ray.remote
def train_model(model_name: str,
                train_config: dict[str, Union[int, list[int]]],
                model_dict: dict[str, Callable] = None,
                model_configs: dict[str, Union[int, float, list[int]]] = None):

    model, optimizer, criterion = setup_model(model_name, model_configs=model_configs, model_dict=model_dict)
    train_loaders, test_loaders = setup_loaders(train_config['permute_seeds'], train_config['batch_size'])
    
    all_task_eval_accuracies = {}
    for task_name, task_loader in train_loaders.items():
        task_accuracies = single_task(model, optimizer, task_loader, task_name, test_loaders, criterion, train_config['epochs_per_task'], device='cpu')
        all_task_eval_accuracies[f'training_{str(task_name)}'] = task_accuracies
    return {'model_name': model_name, 'task_evaluation_acc': all_task_eval_accuracies}


def process_task_accuracies(all_task_accuracies: dict[str, list[float]], epochs_per_task: int, permute_seeds: list[int]) -> tuple[float, float]:
    task0_acc_1 = all_task_accuracies['task_evaluation_acc']['training_None'][None][epochs_per_task-1]
    task0_acc_2 = all_task_accuracies['task_evaluation_acc']['training_42'][None][epochs_per_task-1]
    task42_acc_0 = all_task_accuracies['task_evaluation_acc']['training_None'][42][0]
    task42_acc_1 = all_task_accuracies['task_evaluation_acc']['training_None'][42][epochs_per_task-1]
    remembering = calc_remembering(task0_acc_1, task0_acc_2)
    forward_transfer = calc_forward_transfer(task42_acc_0, task42_acc_1)
    return (remembering, forward_transfer)
    
def get_n_npb(n_brances, n_in):
    return int(n_in/n_brances)

def run_continual_learning(configs: dict[str, Union[int, list[int]]]):
    n_b_1 =  configs['n_b_1'] if 'n_b_1' in configs.keys() else 14
    n_b_2 = n_b_1
    # rotations = configs['rotations'] if 'rotations' in configs.keys() else [0, 180]
    permute_seeds = configs['permute_seeds'] if 'permute_seeds' in configs.keys() else [None]
    epochs_per_task = configs['epochs_per_task']
    batch_size = configs['batch_size']
    MODEL = configs['model_name']
    MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
    MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    MODEL_DICT = {name: model for name, model in zip(MODEL_NAMES, MODEL_CLASSES)}
    TRAIN_CONFIGS = {'batch_size': batch_size,
                    'epochs_per_task': epochs_per_task,
                    'permute_seeds': permute_seeds,}
    
    MODEL_CONFIGS = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': len(TRAIN_CONFIGS['permute_seeds']), 
                    'device': 'cpu', 
                    'n_npb': [get_n_npb(n_b_1, 784), get_n_npb(n_b_2, 784)], 
                    'n_branches': [n_b_1, n_b_2], 
                    'sparsity': configs['sparsity'] if 'sparsity' in configs.keys() else 0.0,
                    'dropout': 0.5,
                    'hidden_layers': [784, 784],
                    'learn_gates': configs['learn_gates'] if 'learn_gates' in configs.keys() else False,
                    }

    if not ray.is_initialized():
        ray.init(num_cpus=70)
    # results = ray.get([train_model.remote(model_name, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS) for model_name in MODEL_NAMES])

    all_task_accuracies = train_model(MODEL, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS)
    remembering, forward_transfer = process_task_accuracies(all_task_accuracies, epochs_per_task, permute_seeds)
    train.report({'remembering': remembering, 'forward_transfer': forward_transfer})
    # print(f'Remembering: {remembering}; Forward Transfer: {forward_transfer}')
    return {'remembering': remembering, 'forward_transfer': forward_transfer}




if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    results = run_continual_learning({'model_name': 'BranchModel', 'n_b_1': 14, 'n_b_2': 14, 'permute_seeds': [None, 42], 'epochs_per_task': 5, 'batch_size': 32})
    print(f'Results: {results}')
