import torch
from torch import nn

from branchNetwork.BranchLayerMM import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from typing import Union
from ipdb import set_trace
import unittest
from time import time
import numpy as np


class BranchModel(nn.Module):
        '''We want the same number of weights for each layer as Masse has.
        layer 1 is 784x2000, layer2 is 2000x2000, layer3 is 2000x10'''
        def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
            super(BranchModel, self).__init__()
            learn_gates = model_configs['learn_gates'] if 'learn_gates' in model_configs else False
            gate_func = model_configs['gate_func'] if 'gate_func' in model_configs else 'sum'
            temp = model_configs['temp'] if 'temp' in model_configs else 1
            # self.layer_1 = nn.Linear(model_configs['n_in'], 2000)
            self.layer_1 = BranchLayer(model_configs['n_in'],
                                      model_configs['n_npb'][0],
                                       model_configs['n_branches'][0],
                                       784, # number of next layer's inputs
                                       device=model_configs['device'],
                                       )
            self.gating_1 = BranchGatingActFunc(784,
                                                model_configs['n_branches'][0],
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'],
                                                learn_gates,
                                                gate_func=gate_func,
                                                temp=temp,
                                                device=model_configs['device'],)
            self.layer_2 = BranchLayer(784,
                                      model_configs['n_npb'][1],
                                       model_configs['n_branches'][1],
                                       784,
                                       device=model_configs['device'])
            self.gating_2 = BranchGatingActFunc(784,
                                                model_configs['n_branches'][1],
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'],
                                                learn_gates,
                                                gate_func=gate_func,
                                                temp=temp,
                                                device=model_configs['device'],
                                                )
            
            self.layer_3 = nn.Linear(784, model_configs['n_out'], device=model_configs['device'])
            self.drop_out = nn.Dropout(model_configs['dropout'])
            self.act_func = nn.ReLU()  
                     
        def forward(self, x, context=0):
            # set_trace()
            x = self.drop_out(self.act_func(self.gating_1(self.layer_1(x), context)))
            x = self.drop_out(self.act_func(self.gating_2(self.layer_2(x), context)))
            return self.layer_3(x)
        
        
def test_Branch(gate_func='sum', temp=1):
    model_configs = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': 5, 
                    'device': 'cpu', 
                    'n_npb': [56, 56], 
                    'n_branches': [1, 1], 
                    'sparsity': 0.8,
                    'dropout': 0,
                    'learn_gates': False,
                    'gate_func': gate_func,
                    'temp': temp}
    
    x = torch.rand(32, 784)
    
    branch_model = BranchModel(model_configs)
    for context in range(model_configs['n_contexts']):
        print(f'Testing context {context}:')
        y = branch_model(x, context)
        assert y.shape == (32, 10)
        print(f"\ttest passed.")
        print(f"\ty sum: {y.sum()}")
        
class TestBranchModelOnGPU(unittest.TestCase):

    def test_gpu_compatibility(self):
        # Setup: Define model configurations assuming proper structure
        model_configs = {
            'n_in': 784,
            'n_out': 10,
            'n_npb': [784, 784],  # Neurons per branch for each layer
            'n_branches': [10, 10],  # Branches for each layer
            'n_contexts': 5,
            'sparsity': 0.1,
            'learn_gates': True,
            'gate_func': 'softmax',
            'temp': 1.0,
            'dropout': 0.5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Check if GPU is available, otherwise use CPU
        }

        # Initialization: Create a BranchModel instance with GPU settings if available
        model = BranchModel(model_configs)
        model.to(model_configs['device'])  # Explicitly move the model to the configured device

        # Create dummy data to feed through the model
        dummy_input = torch.randn(16, 784, device=model_configs['device'])  # Batch size of 16, 784 features

        # Execution: Pass the dummy data through the model
        try:
            output = model(dummy_input)
            print("Output shape:", output.shape)  # Expected output shape (16, 10) for batch size 16 and 10 output features
            print("Output device:", output.device)  # Should print the device used, ideally 'cuda'
        except Exception as e:
            self.fail(f"Model execution on GPU failed with an error: {str(e)}")

        # Assert: Check if the output is on the correct device and has the right shape
        self.assertEqual(output.shape, (16, 10), "Output shape is incorrect.")
        self.assertEqual(output.device.type, model_configs['device'], "Output is not on the correct device.")

class TestBranchModelPerformance(unittest.TestCase):
    def setUp(self):
        self.model_configs = {
            'n_in': 784,
            'n_out': 10,
            'n_npb': [2000, 2000],
            'n_branches': [1, 1],
            'n_contexts': 1,
            'sparsity': 0.1,
            'learn_gates': False,
            'gate_func': 'sum',
            'temp': 1,
            'dropout': 0.5,
            'device': 'cpu'
        }
        batch_size = 512
        self.dummy_input = torch.randn(batch_size, 784)  # Batch size of 64 and input features of 784
        self.target_output = torch.randint(0, 10, (batch_size,))  # Random target outputs for a classification task
        self.num_runs = 1000
        self.loss_function = nn.CrossEntropyLoss()

    def time_model_on_device(self, device):
        model = BranchModel({**self.model_configs, 'device': device}).to(device)
        
        # print([f'{name}: {param}' for name, param in model.named_parameters()])
            
        model.train()

        if device == 'cuda':
            torch.cuda.synchronize(device)
        
        times = []
        for _ in range(self.num_runs):
            input_data = self.dummy_input.to(device)
            target_data = self.target_output.to(device)
            start_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
            end_event = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None

            if start_event:
                start_event.record()

            else:
                start_time = time()
            
            output = model(input_data)
            loss = self.loss_function(output, target_data)
            model.zero_grad()
            loss.backward()

            if end_event:
                end_event.record()
                torch.cuda.synchronize(device)
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert milliseconds to seconds
            else:
                elapsed_time = time() - start_time
            
            times.append(elapsed_time)

        mean_time = np.mean(times)
        std_time = np.std(times)
        return mean_time, std_time

    def test_cpu_performance(self):
        mean_time, std_time = self.time_model_on_device('cpu')
        print(f"CPU: Mean run time per iteration: {mean_time:.6f} sec, Std Dev: {std_time:.6f} sec")

    def test_gpu_performance(self):
        if torch.cuda.is_available():
            mean_time, std_time = self.time_model_on_device('cuda')
            print(f"GPU: Mean run time per iteration: {mean_time:.6f} sec, Std Dev: {std_time:.6f} sec")
        else:
            print("CUDA is not available. Skipping GPU test.")

        
if __name__ == '__main__':
    test_Branch()
    print('BranchModel test passed.')
    for gate_func in ['sum', 'softmax', 'max']:
        for temp in [0.1, 1, 10]:
            test_Branch(gate_func, temp)
            print(f'BranchModel test passed for gate_func: {gate_func} and temp: {temp}')
            
    unittest.main()
