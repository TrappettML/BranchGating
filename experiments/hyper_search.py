from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.dataloader import load_rotated_flattened_mnist_data
from tests.GatingBranchRotatingMNIST import train_epoch, evaluate_model


import torch
import torch.nn as nn
import ray
from ray import tune

from typing import Union


MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
MODEL_CONFIGS = {'n_in': 784, 
                     'n_out': 10, 
                     'n_contexts': 1, 
                     'device': 'cpu', 
                     'n_npb': 200, 
                     'n_branches': 10, 
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
        tune.report(mean_accuracy=accuracy)
    return accuracy

def main():
    ray.init(num_cpus=20)
    analysis = tune.run(
        train_and_evaluate_model,
        config={
            "model_class": tune.grid_search(MODEL_CLASSES),
            "model_configs": MODEL_CONFIGS,
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            "batch_size": tune.grid_search([32, 64, 128]),
            "n_epochs": 20,
        },
        num_samples=10,
        resources_per_trial={"cpu": 2},
    )
    print(analysis.dataframe())
    ray.shutdown()
    
    
if __name__ == "__main__":
    main()