
import torch as th
from torch import nn
from ipdb import set_trace


class BranchLayer(nn.Module):
    def __init__(self, n_in: int, n_npb: int, n_b:int , n_next_h: int, device='cpu') -> None:
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
        self.create_indices()
        self.create_mask()
        self.create_weights()
        assert self.w.shape == (self.n_npb, self.n_b*self.n_next_h), f'weights shape is {self.w.shape} and should be {(self.n_npb, self.n_b*self.n_next_h)}'
        assert self.mask.shape == (self.n_in, self.n_b*self.n_next_h), f'mask shape is {self.mask.shape} and should be {(self.n_in, self.n_b*self.n_next_h)}'
    
    def forward(self, x):
        x = x.to(self.device)
        local_mask = self.mask != 0
        local_weights = self.mask.clone().detach().requires_grad_(False)
        # local_weights[local_mask] = self.w
        local_weights[self.all_branch_indices, th.arange(local_weights.shape[1])] = self.w
        # mask_w = self.mask * self.w
        x = x @ local_weights # Branch Operation (soma operation in gatingActFunction.py)
        x = x.view(-1, self.n_b, self.n_next_h)
        return x
        
    def create_weights(self) -> None:
        # self.w = nn.init.kaiming_uniform_(th.empty(self.n_in, self.n_b*self.n_next_h))
        self.w = nn.init.kaiming_uniform_(th.empty(self.n_npb, self.n_b*self.n_next_h, device=self.device), a=0.1)
        self.w = nn.Parameter(self.w)

    def create_mask(self) -> None:
        self.mask = th.zeros(self.n_in, self.n_b * self.n_next_h, device=self.device)
        col_indices = th.arange(self.n_b * self.n_next_h).repeat(self.n_npb, 1)
        self.mask[self.all_branch_indices, col_indices] = 1    
    
    def create_indices(self) -> None:
        self.all_branch_indices = th.randint(low=0, high=self.n_in, size=(self.n_npb, self.n_b * self.n_next_h), device=self.device, dtype=th.long)

    def _output_shape(self):
        return (self.n_b, self.n_next_h)
    
    def __repr__(self):
        return super().__repr__() + f'\nBranchLayer(n_in={self.n_in}, n_npb={self.n_npb}, n_b={self.n_b}, n_next_h={self.n_next_h})'
    
    
    
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
    # set_trace()
    assert out.shape == (5, 2, 4)
    
if __name__ == '__main__':
    test_branch_layer()
    print("BranchLayer test passed!")