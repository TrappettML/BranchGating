import torch
from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.dataloader import load_rotated_flattened_mnist_data
from branchNetwork.utils.timing import timing_decorator

from torch.utils.data import DataLoader
from torch import nn
from ipdb import set_trace
import ray
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
            if i>2:
                break
            set_trace()
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
        results = ray.get([single_ray_eval.remote(model, images, labels, task, criterion, device=device) for i, (images, labels) in enumerate(data_loader) ]) # if i < 3
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
    results = ray.get([new_ray_evaluate_model.remote(model, task, test_loader, criterion, device=device) for i, (task, test_loader) in enumerate(test_loaders.items())]) # if i < 3
    # results = [ray_evaluate_model(model, task, test_loader, criterion, device=device) for task, test_loader in test_loaders.items()]
    return results # list of dictionaries

def calc_remembering(A_acc_1: float, A_acc_2: float) -> float:
    """Rem = Acc_2 / Acc_1 -> this will yield a value between 0 and 1 (most likely)
       since Acc_2 is after trainging on a new task. If Greater than 1 than we have Backwards Transfer."""
    return A_acc_2/A_acc_1

def calc_forward_transfer(B_acc_0: float, B_acc_1: float) -> float:
    """FT = (B_acc_0 - B_acc_1)/ (B_acc_0 + B_acc_1)
        This will yield a value between -1 and 1. if the value is negative than thre was negative interference.
        if positive than forward transfer of information."""
    return (B_acc_0 - B_acc_1)/ (B_acc_0 + B_acc_1)

def calc_branch_cl_metrics(*args):
    pass
    
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
                model_aggregated_results: OrderedDict['str', Union[str, list[float]]], 
                device:str = 'cpu'):
    # model.train()
    task_accuracies = {}
    print(f'begining training model {model_aggregated_results["model_name"]} on task {train_task}')
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

def setup_loaders(rotation_degrees: list[int], batch_size: int):
    train_loaders = dict()
    test_loaders = dict()
    for degree in rotation_degrees:
        test_loader, train_loader = load_rotated_flattened_mnist_data(batch_size, degree)
        train_loaders[degree] = train_loader
        test_loaders[degree] = test_loader
    return train_loaders, test_loaders

def make_data_container(rotation_degrees: list[int], model_name: str):
    keys = ['model_name', 'Training_Loss']
    for deg in rotation_degrees:
        keys.extend([f'd{deg}_loss', f'd{deg}_Accuracy'])

    values = [model_name, []]
    for _ in rotation_degrees:
        values.extend([[], []])  
        
    model_data = OrderedDict(zip(keys, values))
    return model_data

@timing_decorator
@ray.remote
def train_model(model_name: str,
                train_config: dict[str, Union[int, list[int]]],
                model_dict: dict[str, Callable] = None,
                model_configs: dict[str, Union[int, float, list[int]]] = None):

    model, optimizer, criterion = setup_model(model_name, model_configs=model_configs, model_dict=model_dict)
    train_loaders, test_loaders = setup_loaders(train_config['rotation_degrees'], train_config['batch_size'])
    
    model_data = make_data_container(train_config["rotation_degrees"], model_name)
    all_task_eval_accuracies = {}
    for task_name, task_loader in train_loaders.items():
        task_accuracies = single_task(model, optimizer, task_loader, task_name, test_loaders, criterion, train_config['epochs_per_train'], 
                    model_data, device=device)
        all_task_eval_accuracies[f'training_{str(task_name)}'] = task_accuracies
    return {'model_name': model_name, 'task_evaluation_acc': all_task_eval_accuracies}

def save_results(results, results_path, filename):
    if not os.path.exists(results_path):
        os.makedirs(results_path)    
    with open(f'{results_path}/{filename}.pkl', 'wb') as f:
        dump(results, f)

@timing_decorator
def run_continual_learning():
    # MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
    # MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
    MODEL_CLASSES = [BranchModel]
    MODEL_NAMES = ['BranchModel']
    MODEL_DICT = {name: model for name, model in zip(MODEL_NAMES, MODEL_CLASSES)}
    TRAIN_CONFIGS = {'batch_size': 32,
                    'epochs_per_train': 2,
                    'rotation_degrees': [0, 180],}
    
    MODEL_CONFIGS = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': len(TRAIN_CONFIGS['rotation_degrees']), 
                    'device': 'cpu', 
                    'n_npb': [56, 200], 
                    'n_branches': [14, 10], 
                    'sparsity': 0.8,
                    'dropout': 0.5,
                    'hidden_layers': [2000, 2000],
                    }
    
    if not ray.is_initialized():
        ray.init(num_cpus=70)
    results = ray.get([train_model.remote(model_name, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS) for model_name in MODEL_NAMES])
    ray.shutdown()
    # results = [train_model(model_name, rotation_degrees, epochs_per_train)
    # for model_name in model_names]
    save_results(results, results_path='/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/results', filename='demo_CL_metrics_results')
    print('Results saved')



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    run_continual_learning()
