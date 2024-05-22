# __init__.py
import torch as th
from torch import nn
from ipdb import set_trace

class BranchGatingActFunc(nn.Module):
    def __init__(self, n_next_h, n_branches=1, n_contexts=1, sparsity=0, learn_gates=False):
        '''
        args:
        - n_branches (int): The number of branches.
        - n_next_h (int): The number of neurons in the next hidden layer.
        - n_contexts (int, optional): The number of contexts. Defaults to 1.
        - sparsity (float, optional): The sparsity of the gating function. Defaults to 0, meaning no sparsity, all ones.
                                        value of 1 means fully sparse and results in single 1.
                                    When n_branchs = 1, sparsity will define how many totaly units are 
                                    active in the gating function.  
        Two gating types:
        Masse grating: n_branch = 1, n_contexts > 1, sparsity > 0
        Branch gating: n_branch > 1, n_contexts > 1, sparsity >= 0
        '''
        super(BranchGatingActFunc, self).__init__()
        assert sparsity >= 0 and sparsity <= 1, "Sparsity must be a value between 0 and 1"
        self.sparsity = sparsity
        self.n_next_h = n_next_h
        self.n_contexts = n_contexts
        self.n_branches = n_branches
        self.masks = {}
        self.forward = self.branch_forward if n_branches > 1 else self.masse_forward
        self.seen_contexts = list()
        self.learn_gates = learn_gates
        
    def make_mask(self):
        if self.n_branches == 1: # will be Masse style Model
            mask = generate_interpolated_array(self.n_next_h, self.sparsity)
            return mask.float()
        else:
            return self.gen_branching_mask()
        
    def gen_branching_mask(self):
        return th.stack([generate_interpolated_array(self.n_branches, self.sparsity, self.learn_gates) for _ in range(self.n_next_h)]).float().T
  
    def branch_forward(self, x, context=0):
        '''forward function for when n_b > 1
           sum over the n_b dimension'''
        mask = self.get_context(context)
        return th.sum(x * mask, dim=1)
    
    def masse_forward(self, x, context=0):
        '''forward function for when n_b = 1,
           no sum needed'''
        mask = self.get_context(context)
        return x * mask

    
    def get_context(self, context):
        '''check if context is in seen contexts, and return the index'''
        if context not in self.masks:
            self.masks[context] = self.make_mask()
            assert len(self.masks) <= self.n_contexts, "Contexts are more than the specified number" 
        return self.masks[context]
        
        
            
def generate_interpolated_array(x, sparsity, learn_gates=False):
    """
    Generate a 1D tensor of size x, smoothly transitioning from 1's to 0's
    based on the input sparsity between 0 and 1.

    Parameters:
    - x: The size of the output tensor.
    - sparsity: A single sparsity between 0 and 1 determining the blend of 1's and 0's.

    Returns:
    - A PyTorch tensor according to the specified rules.
    """
    assert x > 0, "x must be greater than 0"
    assert sparsity >= 0, "sparsity must be greater than or equal to 0"
    assert sparsity <= 1, "sparsity must be less than or equal to 1"

    # Calculate the transition index based on the sparsity
    transition_index = round((x - 1) * (1 - sparsity))

    # Create a tensor of 1's and 0's based on the transition index
    output = th.zeros(x)
    output[:transition_index + 1] = 1
    #permute the tensor randomly
    output = output[th.randperm(x)]
    if learn_gates:
        output = make_gates_learnable(output)
    # print(f'output: {output}')
    return output        
    
def make_gates_learnable(gates):
    mask = gates != 0
    values = gates[mask].clone().detach().requires_grad_(True)
    learn_gates = th.zeros_like(gates, dtype=th.float, requires_grad=False)
    learn_gates[mask] = values
    return learn_gates


def test_gating_act_func():
    n_batches = 10
    n_b = 10
    n_next_h = 4
    n_contexts = 2
    for sparsity in [0, 0.5, 1]:
        gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, sparsity)
        x = th.rand(n_batches, n_b, n_next_h)
        out = gating(x)
        assert out.shape == (n_batches, n_next_h), f'Expected shape {(n_batches, n_next_h)}, got {out.shape}'
    print("GatingActFunc test passed")
    
def test_masse_act_func():
    n_batches = 10
    n_b = 1
    n_next_h = 10
    n_contexts = 2
    for sparsity in [0, 0.5, 1]:
        masse_gate = BranchGatingActFunc(n_next_h, n_b, n_contexts, sparsity)   
        x = th.rand(n_batches, n_next_h)
        out = masse_gate(x)
        assert out.shape == (n_batches, n_next_h), f'Expected shape {(n_batches, n_next_h)}, got {out.shape}'
    print("MasseActFunc test passed")
    
def test_learnable_gates():
    n_batches = 10
    n_b = 10
    n_next_h = 4
    n_contexts = 2
    for sparsity in [0, 0.5, 1]:
        gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, sparsity, learn_gates=True)
        x = th.rand(n_batches, n_b, n_next_h)
        out = gating(x)
        assert out.shape == (n_batches, n_next_h), f'Expected shape {(n_batches, n_next_h)}, got {out.shape}'
    print("learnGates test passed")
    
if __name__ == "__main__":
    test_gating_act_func()
    test_masse_act_func()
    test_learnable_gates()