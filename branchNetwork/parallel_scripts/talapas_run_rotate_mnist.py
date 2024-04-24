import torch
import ray
import os
import socket

from branchNetwork.tests.GatingBranchRotatingMNIST import train_model, save_results


def run_continual_learning():
    model_names = ['Masse', 'Simple', 'Branching', 'Expert']
    rotation_degrees = [0, 120, 240]
    epochs_per_train = 40
    ray.init(address='auto')
    results = ray.get([train_model.remote(model_name, rotation_degrees, epochs_per_train) for model_name in model_names])
    # results = [train_model(model_name, rotation_degrees, epochs_per_train)
    # for model_name in model_names]
    save_results(results, results_path='/home/mtrappet/BranchGating/branchNetwork/data/results')
    print('Results saved')
    
if __name__ == '__main__':
    try:
        run_continual_learning()
    except Exception as e:
        print(f'Error: {e}')
        raise e 