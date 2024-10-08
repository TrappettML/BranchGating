
import torch as th
from torch import nn
from ipdb import set_trace
import unittest


class BranchLayer(nn.Module):
    def __init__(self, n_in: int, n_npb: int, n_b:int , n_next_h: int, device='cpu') -> None:
        '''
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
        # set_trace()
        x = x.to(self.device)
        # local_mask = self.mask != 0
        local_weights = self.mask.clone().detach().requires_grad_(False)
        # local_weights[local_mask] = self.w
        local_weights[self.all_branch_indices, th.arange(local_weights.shape[1])] = self.w
        # mask_w = self.mask * self.w
        x = x @ local_weights # Branch Operation (soma operation in gatingActFunction.py)
        x = x.view(-1, self.n_b, self.n_next_h)
        return x
        
    def create_weights(self) -> None:
        # self.w = nn.init.kaiming_uniform_(th.empty(self.n_in, self.n_b*self.n_next_h))
        self.w = nn.init.kaiming_normal_(th.empty(self.n_npb, self.n_b*self.n_next_h, device=self.device))
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
    
class TestBranchLayer(unittest.TestCase):
    def setUp(self):
        # Initialize BranchLayer with some test parameters
        self.layer = BranchLayer(n_in=10, n_npb=5, n_b=3, n_next_h=4, device='cpu')

    def test_average_weights_properties(self):
        magnitudes = []
        variances = []

        for _ in range(10):
            # Re-initialize weights each time to simulate different initializations
            del self.layer.w
            self.layer.create_weights()
            weights = self.layer.w.data
            
            # Calculate magnitude and variance
            magnitude = th.mean(th.abs(weights)).item()
            variance = th.var(weights).item()
            
            # Store results
            magnitudes.append(magnitude)
            variances.append(variance)
        # Convert lists to tensors for variance calculation
        magnitudes_tensor = th.tensor(magnitudes)
        variances_tensor = th.tensor(variances)

        # Calculate average magnitude and variance
        average_magnitude = th.mean(magnitudes_tensor).item()
        average_variance = th.mean(variances_tensor).item()

        # Calculate the variance of magnitudes and the variance of variances
        variance_of_magnitudes = th.var(magnitudes_tensor).item()
        variance_of_variances = th.var(variances_tensor).item()

        print(f"Average magnitude of weights over 10 trials: {average_magnitude}")
        print(f"Average variance of weights over 10 trials: {average_variance}")
        print(f"Variance of magnitudes over 10 trials: {variance_of_magnitudes}")
        print(f"Variance of variances over 10 trials: {variance_of_variances}")


    
if __name__ == '__main__':
    test_branch_layer()
    print("BranchLayer test passed!")
    unittest.main()
    