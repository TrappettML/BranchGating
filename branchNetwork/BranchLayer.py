
import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from ipdb import set_trace


class BranchLayer(nn.Module):
    def __init__(self, n_in, n_npb, n_b, n_next_h, device='cpu') -> None:
        '''
        args:
        - branch_params (dict): A dictionary containing the following:
            - n_in (int): The number of input features.
            - n_npb (int): The number of neurons per branch.
            - n_b (int): The number of branches.
            - n_next_h (int): The number of neurons in the next hidden layer.
            - device (torch.device, optional): The device to use. Defaults to th.device('cpu').
        '''
        super(BranchLayer, self).__init__()
        self.n_in = n_in
        self.n_npb = n_npb
        self.n_b = n_b
        self.n_next_h = n_next_h
        self.device = device
        self.create_all_branch_indices()
        self.create_weights()
        
    def forward(self, x):
        x = self.sample_branches(x) # out shape (n_batches, n_npb, n_b*n_next_h)
        x = self.element_wise_mult(x) # out shape (n_batches, n_npb, n_b*n_next_h)
        x = x.sum(dim=1) # results in shape (n_batches, n_b*n_next_h) # sum of n_npb
        x = x.view(-1, self.n_b, self.n_next_h) # reshape to (n_batches, n_b, n_next_h)
        if self.n_b == 1:
            x = x.squeeze(1)
        return x
        
    def sample_branches(self, x):
        # x is of shape (n_batches, n_in)
        '''This function will sample from x, n_npb times from the n_in dim. 
        It will then repeatedly sample from it n_b*n_next_h times.
        Then it will repeat for each batch'''
        x = x[:, self.all_branch_indices] # results in shape (n_batches, n_npb, n_b*n_next_h)
        return x
        
    def element_wise_mult(self, x):
        return x * self.weights
    
    def create_weights(self) -> None:
        self.w = th.empty(self.n_npb, self.n_b*self.n_next_h)
        nn.init.kaiming_uniform_(self.w)
        self.weights = nn.Parameter(self.w).to(self.device)
        
    def create_all_branch_indices(self) -> None:
        # row is organized as n_b * n_next_h, so iterate 
        # shape of all_branch_indices is (n_npb, n_b * n_next_h)
        self.all_branch_indices = th.randint(low=0, high=self.n_in, size=(self.n_npb, self.n_b * self.n_next_h))
        
    def _output_shape(self):
        return (self.n_b, self.n_next_h)
    
def test_branch_layer():
    branch_params = {
        'n_in': 10,
        'n_npb': 3,
        'n_b': 2,
        'n_next_h': 4
    }
    branch_layer = BranchLayer(**branch_params)
    x = th.randn(5, 10)
    out = branch_layer(x)
    set_trace()
    assert out.shape == (5, 2, 4)
    
if __name__ == '__main__':
    test_branch_layer()
    print("BranchLayer test passed!")