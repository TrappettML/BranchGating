import torch
import ray
import os
import socket
from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel
from branchNetwork.tests.GatingBranchRotatingMNIST import train_model, save_results


def run_continual_learning():
    MODEL_CLASSES = [BranchModel, ExpertModel, MasseModel, SimpleModel]
    MODEL_NAMES = ['BranchModel', 'ExpertModel', 'MasseModel', 'SimpleModel']
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
                    'lr': 0.001,
                    }
    ray.init(address='auto')
    results = ray.get([train_model.remote(model_name, TRAIN_CONFIGS, MODEL_DICT, MODEL_CONFIGS) for model_name in MODEL_NAMES])
    # results = [train_model(model_name, rotation_degrees, epochs_per_train)
    # for model_name in model_names]
    save_results(results, results_path='/home/mtrappet/BranchGating/branchNetwork/data/3task_rotation/results')
    print('Results saved')
    
if __name__ == '__main__':
    try:
        run_continual_learning()
    except Exception as e:
        print(f'Error: {e}')
        raise e 