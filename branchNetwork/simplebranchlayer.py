# __init__.py

import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from ipdb import set_trace

def reshape_and_pad_tensor(input_tensor, p, pad_value=0):
    """
    Reshapes the input tensor from shape (n, m) to (n, ceil(m/p), p), padding with zeros if m is not divisible by p.
    
    Args:
    - input_tensor (torch.Tensor): The input tensor of shape (n, m).
    - p (int): The target size for the last dimension.
    - pad_value (int, optional): Padding value. Defaults to 0.
    
    Returns:
    - torch.Tensor: The reshaped and padded tensor.
    """
    assert len(input_tensor.shape) == 2, "Input tensor must have shape (n, m)"
    n, m = input_tensor.shape
    # Calculate the necessary size of the second dimension after reshaping
    new_second_dim = (m + p - 1) // p  # This ensures that we round up if m is not divisible by p
    
    # Calculate the total number of elements needed after reshaping (and padding if necessary)
    total_elements_needed = n * new_second_dim * p
    # Calculate the number of padding elements to add
    padding = total_elements_needed - n * m
    
    # Reshape the tensor to (n*m), pad it, and then reshape it to the desired output shape
    padded_reshaped_tensor = F.pad(input_tensor.reshape(-1), (0, padding), value=pad_value).reshape(n, new_second_dim, p)
    
    return padded_reshaped_tensor



class BranchLayer(nn.Module):
    def __init__(self, branch_params):
        super(BranchLayer, self).__init__()
        self.n_inputs = branch_params['n_inputs']
        self.branching_factor = branch_params['branching_factor']
        self.device = branch_params['device'] if 'device' in branch_params else th.device('cpu')
        self.n_branches = (self.n_inputs + self.branching_factor - 1) // self.branching_factor # we want 1 + the number of groups
        self.w = th.empty(1, self.n_inputs)
        nn.init.kaiming_uniform_(self.w)
        self.weights = nn.Parameter(self.w)
        # self.branch_layers = nn.ModuleList(self.branch_layers)
        
    def element_wise_mult(self, x):
        return x * self.weights
        
    def forward(self, x):
        x = self.element_wise_mult(x)
        x_reshaped = reshape_and_pad_tensor(x, self.branching_factor)
        x_summed = th.sum(x_reshaped, dim=-1)
        return x_summed
    
    def _output_size(self):
        return self.n_branches
    
    
def test_branch():
    branch_params = {'n_inputs': 11, 'branching_factor': 3}
    example_inputs = th.randn(5, branch_params['n_inputs'])
    branch_layer = BranchLayer(branch_params)  
    print('inputs: ', example_inputs)
    print(branch_layer)
    outs = branch_layer(example_inputs)
    print(outs)
    print(branch_layer._output_size())
    set_trace()
    
if __name__ == '__main__':
    test_branch()   