from branchNetwork.architectures.Branch import BranchModel
from branchNetwork.architectures.Expert import ExpertModel
from branchNetwork.architectures.Masse import MasseModel
from branchNetwork.architectures.Simple import SimpleModel

import torch
import torch.nn as nn
from ipdb import set_trace
import time
import numpy as np

def test_basic_forwards():
    n_batches = 10
    n_in = 784
    n_out = 10
    hidden_layers = [784, 784]
    n_contexts = 2
    n_npb = 56
    n_branches = 14
    sparsity = 0.8
    model_configs = {'n_in': n_in, 
                     'n_out': n_out, 
                     'n_contexts': n_contexts, 
                     'device': 'cpu', 
                     'n_npb': [56, n_npb], 
                     'n_branches': [14, n_branches], 
                     'sparsity': sparsity,
                     'dropout': 0,
                     'learn_gates': True,
                     }
    
    simple_model = SimpleModel(model_configs)
    masse_model = MasseModel(model_configs)
    branch_model = BranchModel(model_configs)
    expert_model = ExpertModel(model_configs)
    
    x = torch.rand(n_batches, n_in)
    y = simple_model(x)
    assert y.shape == (n_batches, n_out)
    print("SimpleModel test passed.")
    
    y = masse_model(x)
    assert y.shape == (n_batches, n_out)
    print("MasseModel test passed.")
    
    y = branch_model(x)
    assert y.shape == (n_batches, n_out)
    print("BranchModel test passed.")
    
    y = expert_model(x)
    assert y.shape == (n_batches, n_out)
    print("ExpertModel test passed.")
    
    
def test_model(model, input, output_shape, context):
    y = model(input, context)
    assert y.shape == output_shape
    print(f"\ttest passed.")
    print(f"\ty sum: {y.sum()}")
    # set_trace()
    y.sum().backward()
    print(f"\tgrads test:{sum([len(p.grad) for p in model.parameters() if p.grad is not None])}")
    
def time_test_model(model, input, output_shape, context):
    start_time = time.time()
    y = model(input, context)
    end_time = time.time()
    assert y.shape == output_shape
    return end_time - start_time

def time_test_multi_contexts():
    n_batches = 10
    n_in = 784
    n_out = 10
    hidden_layers = [2000, 2000]
    n_contexts = 2
    n_npb = 200
    n_branches = 10
    sparsity = 0.8
    model_configs = {'n_in': n_in, 
                     'n_out': n_out, 
                     'n_contexts': n_contexts, 
                     'device': 'cpu', 
                     'n_npb': [56, n_npb], 
                     'n_branches': [14, n_branches], 
                     'sparsity': sparsity,
                     'dropout': 0,
                     'learn_gates': True,
                     }
    
    simple_model = SimpleModel(model_configs)
    masse_model = MasseModel(model_configs)
    branch_model = BranchModel(model_configs)
    expert_model = ExpertModel(model_configs)
    
    x = torch.rand(n_batches, n_in)
    results = {}
    for model in [simple_model, masse_model, branch_model, expert_model]:
        # print(f'Testing model {model.__class__.__name__}')
        times = []
        for _ in range(20):
            for context in range(n_contexts):
                # print(f'Testing context {context}:')
                times.append(time_test_model(model, x, (n_batches, n_out), context))
                results[str(model.__class__.__name__)] = {'mean': np.mean(times), 'std': np.std(times)}
                
    for model_name, stats in results.items():
        print(f'{model_name} - Mean Time: {stats["mean"]:.6f}s, StdDev: {stats["std"]:.6f}s')

if __name__=='__main__':
    test_basic_forwards()
    # test_multi_contexts()
    time_test_multi_contexts()