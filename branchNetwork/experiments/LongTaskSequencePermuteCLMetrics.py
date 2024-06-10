import torch
from branchNetwork.architectures.BranchMM import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.dataloader import load_permuted_flattened_mnist_data
from branchNetwork.dataloader import load_rotated_flattened_mnist_data
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
import socket
import time



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

# @ray.remote
def _evaluate_model(model, task, test_loader, criterion, device='cpu'):
    return {task: evaluate_model(model, task, test_loader, criterion, device=device)}



def get_ray_evaluate_model(device='cpu'):
    print(f'get_ray_evaluate_model: device: {device}')
    if device == 'cuda' or device == torch.device('cuda'):
        @ray.remote(num_gpus=0.125)
        def gpu_evaluate_model(model, task, test_loader, criterion, device=device):
            return {task: evaluate_model(model, task, test_loader, criterion, device=device)}
        return gpu_evaluate_model
    elif device == 'cpu' or device == torch.device('cpu'):
        @ray.remote
        def cpu_evaluate_model(model, task, test_loader, criterion, device=device):
            return {task: evaluate_model(model, task, test_loader, criterion, device=device)}
        return cpu_evaluate_model
 
def evaluate_all_models(model: nn.Module, test_loaders: dict[int, DataLoader], criterion: Callable, device='cpu'):
    if not ray.is_initialized():
            results = [_evaluate_model(model, task, test_loader, criterion, device=device) for task, test_loader in test_loaders.items()]
    else:
        _ray_eval_model = get_ray_evaluate_model(device)
        results = ray.get([_ray_eval_model.remote(model, task, test_loader, criterion, device=device) for task, test_loader in test_loaders.items()])
    return results # list of dictionaries

def calc_remembering(A_acc_1: float, A_acc_2: float) -> float:
    """Rem = Acc_2 / Acc_1 -> this will yield a value between 0 and 1 (most likely)
       since Acc_2 is after trainging on a new task. If Greater than 1 than we have Backwards Transfer.
       General form:
       R^{T_i}_{T_{j} = \frac{a^{T_i}_{T_j}}{a^{T_i}_{T_j-1}}"""
    return A_acc_2/A_acc_1

def calc_forward_transfer(B_acc_0: float, B_acc_1: float) -> float:
    """FT = (B_acc_0 - B_acc_1)/ (B_acc_0 + B_acc_1)
        This will yield a value between -1 and 1. if the value is negative than negative interference.
        if positive than forward transfer of information.
        General form:
        FT^{T_i}_{T_{j} = \frac{a^{T_i}_{T_{j-1}} - a^{T_i}_{T_{j}}}{a^{T_i}_{T_j} + a^{T_i}_{T_{j-1}}}"""
    return (B_acc_1 - B_acc_0)/ (B_acc_0 + B_acc_1)

    
def gather_task_accuracies(all_test_results: dict[str, list[float]], test_results: list[dict[int, tuple[float, float]]]) -> list[float]:
    for result in test_results:
        for task, (acc, _) in result.items():
            all_test_results.setdefault(task, []).append(acc)
    return all_test_results

def get_first_last_accuracies(all_eval_accuracies: dict[str, list[float]]) -> list[dict[str, float]]:
    first_last_dict = []
    for task, accs in all_eval_accuracies.items():
        eval_task_dict = dict()
        eval_task_dict['task_name'] = task
        eval_task_dict['first_acc'] = accs[0]
        eval_task_dict['last_acc'] = accs[-1]
        first_last_dict.append(eval_task_dict)
    return first_last_dict

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
        train_loss = train_epoch(model, train_loader, train_task, optimizer, criterion, device=device)
        test_results = evaluate_all_models(model, test_loaders, criterion, device=device)
        task_accuracies = gather_task_accuracies(task_accuracies, test_results)
    eval_first_last_accuracies = get_first_last_accuracies(task_accuracies)
    single_task_data = {'train_task': train_task, 'eval_accuracies': eval_first_last_accuracies}
    # set_trace()
    return task_accuracies, single_task_data

def setup_model(model_name: str, 
                model_configs: dict[str, Union[int, list[int], dict[str, int], str]], 
                model_dict: Union[None, dict[str, Callable]]):
    
    assert model_name in model_dict.keys(), f'{model_name} not in model_dict'
    model = model_dict[model_name](model_configs)   
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optim, criterion

def setup_loaders(rotation_degrees: list[int], batch_size: int, n_test_tasks: int = 3):
    test_tasks = rotation_degrees[:n_test_tasks]
    train_loaders = dict()
    test_loaders = dict()
    for angle in rotation_degrees:
        # test_loader, train_loader = load_permuted_flattened_mnist_data(batch_size, angle)
        test_loader, train_loader = load_rotated_flattened_mnist_data(batch_size, rotation_in_degrees=angle)
        train_loaders[angle] = train_loader
        if angle in test_tasks:
            test_loaders[angle] = test_loader
    return train_loaders, test_loaders

# @timing_decorator
# @ray.remote
def train_model(model_name: str,
                train_config: dict[str, Union[int, list[int]]],
                model_dict: dict[str, Callable] = None,
                model_configs: dict[str, Union[int, float, list[int]]] = None):

    model, optimizer, criterion = setup_model(model_name, model_configs=model_configs, model_dict=model_dict)
    train_loaders, test_loaders = setup_loaders(train_config['rotation_degrees'], train_config['batch_size'], train_config['n_test_tasks'])
    
    all_task_eval_accuracies = {}
    all_first_last_data = []
    for task_name, task_loader in train_loaders.items():
        task_accuracies, first_last_data = single_task(model, optimizer, task_loader, task_name, test_loaders, criterion, train_config['epochs_per_task'], device=model_configs['device'])
        all_task_eval_accuracies[f'training_{str(task_name)}'] = task_accuracies
        all_first_last_data.append(first_last_data)
        print(f'\n\n______________Finished training on task {task_name}______________\n\n')
    # set_trace()
    return {'model_name': model_name, 'task_evaluation_acc': all_task_eval_accuracies, 'first_last_data': all_first_last_data}


def calc_metrics(d_j, prev_d_j, train_task):
    task_metrics = []
    for metric_dict in d_j:
        for prev_metric_dict in prev_d_j:
            if metric_dict['task_name'] == prev_metric_dict['task_name']:
                prev_d_j_metric = prev_metric_dict['last_acc']
        eval_task = metric_dict['task_name']
        # set_trace()
        rem = calc_remembering(prev_d_j_metric, metric_dict['last_acc'] )
        ft = calc_forward_transfer(metric_dict['last_acc'], prev_d_j_metric)
        if eval_task == train_task:
            rem = None
            rem = None
        task_metrics.append({'task_name': eval_task, 'remembering': rem, 'forward_transfer': ft})
    return task_metrics

def process_all_sequence_metrics(first_last_data: list[dict[str, float]]) -> list[dict[str, Union[str, dict[list]]]]:
    sequence_results = []
    # zeroth task - calc F.T. but not remembering
    first_train_name = first_last_data[0]['train_task']
    # k is train_task /name
    # v is eval_accuracies
    zero_task_metrics = []
    for eval_task in first_last_data[0]['eval_accuracies']:
        if eval_task['task_name'] != first_train_name:
            rem = None
            ft = calc_forward_transfer(eval_task['first_acc'], eval_task['last_acc'])
        else:
            rem = None
            ft = None
        zero_task_metrics.append({'task_name': eval_task['task_name'], 'remembering': rem, 'forward_transfer': ft})
    sequence_results.append({'train_task': first_train_name, 'metrics': zero_task_metrics})
    for i in range(1, len(first_last_data)):
        train_task = first_last_data[i]['train_task']
        prev_d_j = first_last_data[i-1]['eval_accuracies']
        d_j = first_last_data[i]['eval_accuracies']
        t_metrics = calc_metrics(d_j, prev_d_j, train_task)
        sequence_results.append({'train_task': train_task, 'metrics': t_metrics})
    # set_trace()
    return sequence_results


def get_n_npb(n_brances, n_in):
    return int(n_in/n_brances)

def pickle_data(data_to_be_saved, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(f'{file_path}/{file_name}.pkl', 'wb') as f:
        dump(data_to_be_saved, f)
    print(f'Saved results to {file_path}/{file_name}.pkl')


def run_continual_learning(configs: dict[str, Union[int, list[int]]]):
    n_b_1 =  configs['n_b_1'] if 'n_b_1' in configs.keys() else 14
    n_b_2 = n_b_1
    # rotations = configs['rotations'] if 'rotations' in configs.keys() else [0, 180]
    rotation_degrees = configs['rotation_degrees'] if 'rotation_degrees' in configs.keys() else [0]
    epochs_per_task = configs['epochs_per_task']
    batch_size = configs['batch_size']
    MODEL = configs['model_name']
    MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
    MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    MODEL_DICT = {name: model for name, model in zip(MODEL_NAMES, MODEL_CLASSES)}
    TRAIN_CONFIGS = {'batch_size': batch_size,
                    'epochs_per_task': epochs_per_task,
                    'rotation_degrees': rotation_degrees,
                    'n_test_tasks': 3,
                    'file_path': 'branchNetwork/data/longsequence/' if 'file_path' not in configs else configs['file_path'],
                    'file_name': 'longsequence_data' if 'file_name' not in configs else configs['file_name']}
    
    MODEL_CONFIGS = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': len(TRAIN_CONFIGS['rotation_degrees']), 
                    'device': 'cpu' if 'device' not in configs else configs['device'], 
                    'n_npb': [get_n_npb(n_b_1, 784), get_n_npb(n_b_2, 784)], 
                    'n_branches': [n_b_1, n_b_2], 
                    'sparsity': configs['sparsity'] if 'sparsity' in configs.keys() else 0.0,
                    'dropout': 0.5,
                    'hidden_layers': [784, 784],
                    'learn_gates': configs['learn_gates'] if 'learn_gates' in configs.keys() else False,
                    'gate_func': configs['gate_func'] if 'gate_func' in configs.keys() else 'sum',
                    'temp': configs['temp'] if 'temp' in configs.keys() else 1.0,
                    }

    if not ray.is_initialized():
        if configs['device'] == 'cuda':
            gpus = torch.cuda.device_count()
            ray.init(num_cpus=5, num_gpus=gpus)
        else:
            ray.init(num_cpus=5)
            
    # results = ray.get([train_model.remote(model_name, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS) for model_name in MODEL_NAMES])

    all_task_accuracies = train_model(MODEL, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS)
    sequence_metrics = process_all_sequence_metrics(all_task_accuracies['first_last_data'])
    # train.report({'remembering': remembering, 'forward_transfer': forward_transfer})
    # print(f'Remembering: {remembering}; Forward Transfer: {forward_transfer}')
    # pickle the results
    pickle_data(sequence_metrics, TRAIN_CONFIGS['file_path'], f'sequential_eval_task_metrics_{MODEL}_sparsity_{MODEL_CONFIGS["sparsity"]}_n_b_1_{n_b_1}_learn_gates_{MODEL_CONFIGS["learn_gates"]}_repeat_{configs["repeat"]}')
    pickle_data(all_task_accuracies, TRAIN_CONFIGS['file_path'], f'all_task_accuracies_{MODEL}_sparsity_{MODEL_CONFIGS["sparsity"]}_n_b_1_{n_b_1}_learn_gates_{MODEL_CONFIGS["learn_gates"]}_repeat_{configs["repeat"]}')
    print(f'Finished training {MODEL} with sparsity {MODEL_CONFIGS["sparsity"]}, n_b_1 {n_b_1}, learn_gates {MODEL_CONFIGS["learn_gates"]}')



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    angle_increments = 60
    time_start = time.time()
    results = run_continual_learning({'model_name': 'BranchModel', 'n_b_1': 14, 'n_b_2': 14, 'rotation_degrees': [int(i) for i in range(0, 360, int(360/angle_increments))], 
                                      'epochs_per_task': 2, 'batch_size': 32, 'gate_func': 'median', 'temp': 1.0, 'device': device})
    time_end = time.time()
    print(f'Time to complete: {time_end - time_start}')
    # print(f'Results: {results}')
    print("________Finished_________")
