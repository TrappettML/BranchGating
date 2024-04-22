
import torch
from branchNetwork.simpleMLP import SimpleMLP
from branchNetwork.BranchLayer import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from branchNetwork.dataloader import load_rotated_flattened_mnist_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm
from ipdb import set_trace
from matplotlib.ticker import MaxNLocator
import ray
from collections import OrderedDict
from typing import Callable, Union
from pickle import dump

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
            if i>10:
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
        results = ray.get([single_ray_eval.remote(model, images, labels, task, criterion, device=device) for i, (images, labels) in enumerate(data_loader)])
        for correct_batch, loss_batch, total_batch in results:
            total += total_batch
            correct += correct_batch
            total_loss += loss_batch
    accuracy = 100 * correct / total
    return accuracy, total_loss / len(data_loader)

@ray.remote
def new_ray_evaluate_model(model, task, test_loader, criterion, device='cpu'):
    return {task: parallel_eval_data(model, task, test_loader, criterion, device=device)}

# @ray.remote
def ray_evaluate_model(model, task, test_loader, criterion, device='cpu'):
    return {task: evaluate_model(model, task, test_loader, criterion, device=device)}
     
def parallel_evaluate_model(model: nn.Module, test_loaders: dict[int, DataLoader], criterion: Callable, device='cpu'):
    results = ray.get([new_ray_evaluate_model.remote(model, task, test_loader, criterion, device=device) for i, (task, test_loader) in enumerate(test_loaders.items()) if i < 2])
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
   
def make_plot(results_dictionary: dict[str, OrderedDict], subfig_labels: list, results_indices: list, title: str, save_str: str, yaxis_label: str='Loss'):
    fig, axs = plt.subplots(len(subfig_labels), 1, figsize=(10, 8), sharex=True, layout="constrained")
    colors ={model_name:color for model_name, color in zip(results_dictionary.keys(), ['blue', 'orange', 'green', 'red'])}
    
    for j, (i, ax) in enumerate(zip(results_indices, axs)):
        for results in results_dictionary.values():
            ax.plot(np.array(list(results.values())[i]), 
                    label=list(results.values())[0], 
                    color=colors[list(results.values())[0]], 
                    linewidth=2)
        
        # Removing the red rectangle by not adding it this time
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.set_yticks([0, 1]) 
        
        # Adding vertical bars
        third_of_data = int(len(list(results.values())[i])/3)
        ax.axvline(x=third_of_data, color='grey', linestyle='--', linewidth=2)
        ax.axvline(x=2*third_of_data, color='grey', linestyle='--', linewidth=2)
        
        # Setting larger labels
        ax.set_ylabel(yaxis_label, fontsize=14)

        # Placing larger, bold text aligned with each subplot on the y-axis
        fig.text(1.05, 0.5-(i*0.01), subfig_labels[j], fontsize=12, fontweight='bold',
                verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)

    # Setting the x-axis label only once, with a larger font size
    axs[-1].set_xlabel('Epochs', fontsize=14)

    # Adding a legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Adding an overall figure title, a bit larger
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(f'/home/users/MTrappett/mtrl/BranchGatingProject/data/plots/{save_str}.png', bbox_inches='tight')
      
def plot_results(results_dictionary: dict[str, OrderedDict]):
    '''results dictionary will have keys of model name and values of Ordereddict with element:
    ['model_name',
    'Training_Loss', 
    f'd{degrees[0]}_loss', 
    f'd{degrees[1]}_loss',
    f'd{degrees[2]}_loss',
    f'd{degrees[0]}_Accuracy',
    f'd{degrees[1]}_Accuracy',
    f'd{degrees[2]}_Accuracy',]
    '''
    
    subfig_labels = ['Training Loss',
                    'Validation Loss for Task 1', 
                    'Validation Loss for Task 2', 
                    'Validation Loss for Task 3']
    results_indices = [1,2,3,4]
    make_plot(results_dictionary, subfig_labels, results_indices, 'Comparison of all losses during training and evaluation', 'loss_plot', 'Loss')
    
    subfig_labels = ['Training Loss',
                    'Validation Accuracy for Task 1', 
                    'Validation Accuracy for Task 2', 
                    'Validation Accuracy for Task 3']
    results_indices = [1,5,6,7]
    make_plot(results_dictionary, subfig_labels, results_indices,'Comparison of evaluation accuracy after training', 'accuracy_plot', 'Accuracy')

def single_task(model:nn.Module,
                optimizer: Callable, 
                train_loader: DataLoader, 
                train_task: Union[str, int],
                test_loaders: dict[int, DataLoader], 
                criterion: Callable, 
                epochs: int,
                model_aggregated_results: OrderedDict['str', Union[str, list[float]]], 
                device:str = 'cpu'):
    model.train()
    print(f'begining training for task {train_task} on model: {model_aggregated_results["model_name"]}')
    for epoch in range(epochs):
        train_losses = train_epoch(model, train_loader, train_task, optimizer, criterion, device=device)
        test_results = parallel_evaluate_model(model, test_loaders, criterion, device=device)
        merge_results_for_model(model_aggregated_results, train_losses, test_results, train_task)
    return model_aggregated_results

def setup_expert(model_configs: dict[str, Union[int, list[int], dict[str, int], str]]):
    class ExpertMLP(SimpleMLP):
        def __init__(self):
            super(ExpertMLP, self).__init__(model_configs['n_in'], 
                      model_configs['hidden_layers'], 
                      model_configs['n_out'],)
            self.seen_contexts = list()
            self.n_contexts = model_configs['n_contexts']
            self.models = [setup_simple(model_configs) for _ in range(model_configs['n_contexts'])]
            self.current_model = self.models[0]
            
        def forward(self, x, context=0):
            self.check_context(context)
            return self.current_model(x)
        
        def check_context(self, context):
            '''check if it is a new context, if it is, switch to that model'''
            if context not in self.seen_contexts:
                self.seen_contexts.append(context)
                assert len(self.seen_contexts) <= self.n_contexts, "Contexts are more than the specified number" 
                self.current_model = self.models[self.seen_contexts.index(context)]
                
    return ExpertMLP()
            
def setup_masse(model_configs: dict[str, Union[int, list[int], dict[str, int], str]]):
    class MasseMLP(nn.Module):
        def __init__(self):
            super(MasseMLP, self).__init__()
            self.layer_0 = SimpleMLP(model_configs['n_in'],
                                     [],
                                     2000)
            self.layer_1 = BranchLayer(2000, 
                                       2000, 
                                       1, 
                                       2000,
                                       model_configs['device'])
            self.layer_2 = BranchLayer(2000,
                                       2000,
                                       1,
                                       model_configs['n_out'],
                                       device=model_configs['device'])
            self.gating_1 = BranchGatingActFunc(2000,
                                                1,
                                                model_configs['n_contexts'],
                                                0.8)
            self.gating_2 = BranchGatingActFunc(2000,
                                                1,
                                                model_configs['n_contexts'],
                                                0.8)
            self.act_func = nn.ReLU()   
                    
        def forward(self, x, context=0):
            x = self.act_func(self.gating_1(self.layer_0(x), context))
            x = self.act_func(self.gating_2(self.layer_1(x), context))
            return self.layer_2(x)
        
    return MasseMLP()

def setup_branching(model_configs: dict[str, Union[int, list[int], dict[str, int], str]]):
    class BranchMLP(nn.Module):
        '''We want the same number of weights for each layer as Masse has.
        layer 1 is 784x2000, layer2 is 2000x2000, layer3 is 2000x10'''
        def __init__(self):
            super(BranchMLP, self).__init__()
            self.layer_1 = SimpleMLP(model_configs['n_in'],
                                     [], 2000)
            self.layer_2 = BranchLayer(2000,
                                      200,
                                       10,
                                       2000,
                                       device=model_configs['device'])
            self.layer_3 = SimpleMLP(2000,[], 10)
            self.gating_1 = BranchGatingActFunc(2000,
                                                10,
                                                model_configs['n_contexts'],
                                                0.8)
            # self.gating_2 = BranchGatingActFunc(model_configs['n_next_h'],
            #                                     model_configs['n_branches'],
            #                                     model_configs['n_contexts'],
            #                                     0.8)
            self.act_func = nn.ReLU()           
        def forward(self, x, context=0):
            x = self.act_func(self.layer_1(x))
            x = self.act_func(self.gating_1(self.layer_2(x), context))
            return self.layer_3(x)
    return BranchMLP()

def setup_simple(model_configs: dict[str, Union[int, list[int], dict[str, int], str]]):
    model = SimpleMLP(model_configs['n_in'], 
                      model_configs['hidden_layers'], 
                      model_configs['n_out'],)
    return model

def setup_model(model_name: str, 
                model_configs: dict[str, Union[int, list[int], dict[str, int], str]]):
    if model_name == 'Masse':
        model = setup_masse(model_configs)
    elif model_name == 'Simple':
        model = setup_simple(model_configs)
    elif model_name == 'Branching':
        model = setup_branching(model_configs)
    elif model_name == 'Expert':
        model = setup_expert(model_configs)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optim, criterion


def setup_loaders(rotation_degrees):
    train_loaders = dict()
    test_loaders = dict()
    for degree in rotation_degrees:
        test_loader, train_loader = load_rotated_flattened_mnist_data(32, degree)
        train_loaders[degree] = train_loader
        test_loaders[degree] = test_loader
    return train_loaders, test_loaders

@ray.remote
def train_model(model_name: str,
                rotation_degrees: list[int], 
                epochs: int,
                device: str = 'cpu'):
    model_config = dict(
        n_in=784,
        hidden_layers=[2000, 2000],
        n_out=10,
        device='cpu',
        n_contexts=3,
    )
    model, optimizer, criterion = setup_model(model_name, model_configs=model_config)
    train_loaders, test_loaders = setup_loaders(rotation_degrees)
    
    model_data = OrderedDict(zip(['model_name','Training_Loss', f'd{rotation_degrees[0]}_loss', 
                                  f'd{rotation_degrees[1]}_loss',f'd{rotation_degrees[2]}_loss',
                                  f'd{rotation_degrees[0]}_Accuracy',f'd{rotation_degrees[1]}_Accuracy',
                                  f'd{rotation_degrees[2]}_Accuracy',], [model_name, [], [], [], [], [], [], []]))
    for task_name, task_loader in train_loaders.items():
        single_task(model, optimizer, task_loader, task_name, test_loaders, criterion, epochs, 
                    model_data, device=device)
    return {'model_name': model_name, 'data': model_data}

def process_and_plot_results(results):
    results_dictionary = dict()
    for result in results:
        results_dictionary[result['model_name']] = result['data']
    plot_results(results_dictionary)
    
def save_results(results):
    with open('/home/users/MTrappett/mtrl/BranchGatingProject/data/results/results.pkl', 'wb') as f:
        dump(results, f)
        
def run_continual_learning():
    model_names = ['Masse', 'Simple', 'Branching', 'Expert']
    rotation_degrees = [0, 120, 240]
    epochs_per_train = 3
    ray.init(num_cpus=70)
    results = ray.get([train_model.remote(model_name, rotation_degrees, epochs_per_train) for model_name in model_names])
    ray.shutdown()
    # results = [train_model(model_name, rotation_degrees, epochs_per_train)
    # for model_name in model_names]
    save_results(results)
    # process_and_plot_results(results)



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.')
    run_continual_learning()