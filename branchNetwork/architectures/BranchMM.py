import torch
from torch import nn

from branchNetwork.BranchLayerMM import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from typing import Union
from ipdb import set_trace
import unittest
from time import time
import numpy as np
import os
import inspect



class BranchModel(nn.Module):
        '''
        layer 1 is 784x784, layer2 is 784x784, layer3 is 784x10'''
        def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
            super(BranchModel, self).__init__()
            # set_trace()
            learn_gates = model_configs['learn_gates'] if 'learn_gates' in model_configs else False
            soma_func = model_configs['soma_func'] if 'soma_func' in model_configs else 'sum'
            # self.layer_1 = nn.Linear(model_configs['n_in'], 784)
            layer_2_n_in = 784 if 'hidden' not in model_configs else model_configs['hidden'][0]
            layer_3_n_in = 784 if 'hidden' not in model_configs else model_configs['hidden'][1]
            drop_ratio = model_configs.get('dropout', 0.0)
            
            self.layer_1 = BranchLayer(n_in=model_configs['n_in'],
                                      n_npb=model_configs['n_npb'][0],
                                       n_b = model_configs['n_branches'][0],
                                       n_next_h=layer_2_n_in, # number of next layer's inputs
                                       device=model_configs['device'],
                                       )
            self.gating_1 = BranchGatingActFunc(layer_2_n_in,
                                                model_configs['n_branches'][0],
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'],
                                                learn_gates,
                                                soma_func=soma_func,
                                                device=model_configs['device'],
                                                det_masks=model_configs.get('det_masks', False),)
            self.layer_2 = BranchLayer(layer_2_n_in,
                                      model_configs['n_npb'][1],
                                       model_configs['n_branches'][1],
                                       layer_3_n_in,
                                       device=model_configs['device'])
            self.gating_2 = BranchGatingActFunc(layer_3_n_in,
                                                model_configs['n_branches'][1],
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'],
                                                learn_gates,
                                                soma_func=soma_func,
                                                device=model_configs['device'],
                                                det_masks=model_configs.get('det_masks', False),)
            
            self.layer_3 = nn.Linear(layer_3_n_in, model_configs['n_out'], device=model_configs['device'])
            self.drop_out = nn.Dropout(drop_ratio)
            self.act_func = model_configs.get('act_func', nn.ReLU)()
            
                     
        def forward(self, x, context=0):
            # set_trace()
            self.branch_activities_1 = self.layer_1(x)
            self.soma_activities_1 = self.gating_1(self.branch_activities_1, context)
            self.gated_branch_1 = self.gating_1.gated_branches
            x = self.drop_out(self.act_func(self.soma_activities_1))
            self.x1_hidden = x
            self.branch_activities_2 = self.layer_2(x)
            self.soma_activities_2 = self.gating_2(self.branch_activities_2, context)
            self.gated_branch_2 = self.gating_2.gated_branches
            x = self.drop_out(self.act_func(self.soma_activities_2))
            self.x2_hidden = x
            return self.layer_3(x)
        
        def heb_branch_update(self, eta=0.1):
            pass

        # def __repr__(self):
            
        #     return super().__repr__() + f'\n path: {os.path.abspath(inspect.getfile(self.__class__))}'
        
        
def test_Branch(soma_func='sum'):
    model_configs = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': 5, 
                    'device': 'cpu', 
                    'n_npb': [56, 56], 
                    'n_branches': [1, 1], 
                    'sparsity': 0.8,
                    'dropout': 0,
                    'learn_gates': False,
                    'soma_func': soma_func,
                    }
    
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
            'soma_func': 'softmax_1.0',
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
            'soma_func': 'sum',
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

def test_branch_model_on_varied_configurations(soma_func='sum'):
    # Define base model configuration
    base_config = {
        'n_in': 784, 
        'n_out': 10, 
        'n_contexts': 5, 
        'device': 'cpu', 
        'n_npb': [56, 56],  # Assumed to stay constant for this test
        'n_branches': [1, 1],  # This will vary
        'sparsity': 0.8,  # This will vary
        'dropout': 0,
        'learn_gates': False,
        'soma_func': soma_func,
    }

    # Test settings for number of branches and sparsity values
    branch_counts = [[1, 1], [2, 2], [7,7]]  # Example branch configurations
    sparsity_values = [0.5, 0.7, 0.9]  # Example sparsity configurations

    # Random input tensor
    x = torch.rand(32, 784)

    # Iterating over each configuration
    for branches in branch_counts:
        for sparsity in sparsity_values:
            print(f"Testing with {branches} branches and {sparsity} sparsity:")
            # Update the model configuration for this test iteration
            config = base_config.copy()
            config['n_branches'] = branches
            config['sparsity'] = sparsity
            
            # Create the model with the current configuration
            branch_model = BranchModel(config)
            
            # Run the model for each context
            for context in range(config['n_contexts']):
                print(f'\tTesting context {context}:')
                y = branch_model(x, context)
                assert y.shape == (32, 10), "Output shape is incorrect"
                print(f"\tTest passed. Output shape: {y.shape}")
                print(f"\tOutput sum for this configuration: {y.sum().item()}")
 
def test_det_branch_model():  
    import pprint            
    def _test_branch_model(sparsity_values, context_values, input_tensor):
        model_configs = {
            'n_in': 784,
            'n_out': 10,
            'n_npb': [int(784/7), int(784/7)],
            'n_branches': [7, 7],
            'hidden_sizes': [784, 784],
            'device': 'cpu',
            'dropout': 0.5,
            'learn_gates': False,
            'soma_func': 'sum',
            'det_masks': True,
            'n_contexts': 12,
        }

        outputs = {}
        for sparsity in sparsity_values:
            model_configs['sparsity'] = sparsity
            model = BranchModel(model_configs)
            for context in context_values:
                set_trace()
                output = model(input_tensor, context)
                outputs[(sparsity, context)] = np.mean(output.detach().numpy())

        return outputs

    # Define sparsity and context ranges
    sparsity_values = np.arange(0, 1, 0.1)
    context_values = np.arange(0, 360, 30)

    # Create a random input tensor
    input_tensor = torch.randn(1, 784)  # Example size for a single input

    # Test the model
    outputs = _test_branch_model(sparsity_values, context_values, input_tensor)
    print("Outputs for different sparsity and context values:")
    pprint.pprint(outputs)

if __name__ == '__main__':
    # test_Branch()
    # print('BranchModel test passed.')
    # for soma_func in ['sum', 'softmax', 'max']:
    #     for temp in [0.1, 1, 10]:
    #         if 'soft' in soma_func:
    #             soma_func = soma_func + '_' + str(temp)
    #         test_Branch(soma_func)
    #         print(f'BranchModel test passed for soma_func: {soma_func} and temp: {temp}')
    # test_branch_model_on_varied_configurations()
    # unittest.main()
    test_det_branch_model()