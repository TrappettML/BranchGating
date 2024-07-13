# __init__.py

import torch 
from torch import nn
from ipdb import set_trace
import re
import unittest

class BranchGatingActFunc(nn.Module):
    def __init__(self, n_next_h, n_b=1, n_contexts=1, sparsity=0, learn_gates=False, soma_func='sum', device='cpu'):
        '''
        args:
        - n_b (int): The number of branches.
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
        self.n_b = n_b
        self.masks = {}
        self.forward = self.branch_forward if n_b > 1 else self.masse_forward
        self.get_context = self.get_learning_context if learn_gates else self.get_unlearning_context
        self.seen_contexts = list()
        self.learn_gates = learn_gates
        self.current_context = None
        self.soma_act_func = self.set_soma_func(soma_func)
        self.device = device
        if learn_gates:
            self.make_learnable_parameters()
            self.all_grads_false = False
            self.activate_current_context = False
    
    def set_soma_func(self, soma_func):
        if soma_func == 'sum':
            def my_sum(x):
                return torch.sum(x, dim=1)
            return my_sum
        elif soma_func == 'median':
            def my_median(x):
                return torch.median(x, dim=1)[0]
            return my_median
        elif soma_func == 'max':
            def my_max(x):
                return torch.max(x, dim=1)[0]
            return my_max
        elif 'softmaxsum' in soma_func.lower():
            pattern = r'softmaxsum_([\d.]+)'
            match = re.search(pattern, soma_func)
            temp = float(match.group(1))
            def my_smx_sum(x):
                sm = nn.functional.softmax(x/temp, dim=1)
                sm_mm = x * sm
                return torch.sum(sm_mm, dim=1)
            return my_smx_sum
        elif 'softmax' in soma_func.lower():
            pattern = r'softmax_([\d.]+)'
            match = re.search(pattern, soma_func)
            temp = float(match.group(1))
            def my_softmax(x):
                sm = nn.functional.softmax(x/temp, dim=1)
                sm_2d = sm.view(-1, sm.size(1))
                sampled_indices = torch.multinomial(sm_2d, num_samples=1)
                sampled_indices = sampled_indices.view(x.size(0), x.size(2)) 
                batch_indices = torch.arange(x.size(0)).unsqueeze(-1).expand_as(sampled_indices)
                max_x = x[batch_indices, sampled_indices, torch.arange(x.size(2))]
                return max_x
            return my_softmax
        elif 'lse' in soma_func.lower():
            pattern = r'lse_([\d.]+)'
            match = re.search(pattern, soma_func)
            temp = float(match.group(1))
            def my_lse(x):
                ones = torch.ones(x.size(0), 1, x.size(2), device=x.device)
                x = torch.cat((x, ones), dim=1)
                return torch.logsumexp(x/temp, dim=1)
            return my_lse
        else:
            raise ValueError(f"soma_func must be one of ['sum', 'max', 'softmax', 'softmax_sum', 'lse'], got {soma_func}")
        
    def make_learnable_parameters(self):
        self.learnable_parameters = nn.ParameterDict()
        for i in range(self.n_contexts):
            self.masks[str(i)] = self.gen_branching_mask()
            self.learnable_parameters['unsigned' + str(i)] = self.make_learnable_gates(self.masks[str(i)])
        
    # def make_mask(self):
    #     if self.n_b == 1: # will be Masse style Model
    #         mask = self.generate_interpolated_array(self.n_next_h)
    #         return mask.float()
    #     else:
    #         return self.gen_branching_mask()
        
    def _gen_branching_mask(self):
        return torch.stack([self.generate_interpolated_array(self.n_b) for _ in range(self.n_next_h)]).float().T
    
    def gen_branching_mask(self):
        # Check if n_b is 1 to switch sparsity along n_next_h dimension
        if self.n_b == 1:
            # Generate a mask with sparsity along the n_next_h dimension
            t_i = self.get_transition_index(self.n_next_h)
            mask = torch.zeros(self.n_next_h, dtype=torch.float32, device=self.device)
            mask[:t_i] = 1
            random_indices = torch.argsort(torch.rand(self.n_next_h, device=self.device))
            mask = mask[random_indices]
            mask = mask.unsqueeze(0)  # Add a singleton dimension for compatibility
        else:
            # Generate a mask with sparsity along the n_b dimension
            t_i = self.get_transition_index(self.n_b)
            mask = torch.zeros(self.n_b, self.n_next_h, dtype=torch.float32, device=self.device)
            mask[:t_i, :] = 1
            random_indices = torch.argsort(torch.rand(self.n_b, self.n_next_h, device=self.device), dim=0)
            mask = torch.gather(mask, 0, random_indices)
        return mask
        
    def branch_forward(self, x, context=0):
        x = x.to(self.device)
        '''forward function for when n_b > 1
           sum over the n_b dimension'''
        # set_trace()
        context = str(context)
        gate = self.get_context(context)
        # return torch.sum(x * gate, dim=1) # x is shape (n_batches, n_b, n_next_h), and so we are summing over branches. 
        return self.soma_act_func(x * gate)
    
    def masse_forward(self, x, context=0):
        x = x.to(self.device)
        '''forward function for when n_b = 1,
           no sum needed'''
        context = str(context)
        gate = self.get_context(context)
        out = x * gate
        if len(out.shape) == 3:
            assert out.shape[1] == 1, "Expected n_b to be 1"
            out = out.squeeze(1)
        return out

    def get_unlearning_context(self, context):
        '''check if context is in seen contexts, and return the index'''
        if context not in self.masks:
            self.masks[context] = self.gen_branching_mask()
            assert len(self.masks) <= self.n_contexts, "Contexts are more than the specified number" 
        return self.masks[context]
    
    def get_learning_context(self, context):
        '''check if context is in seen contexts, and return the index
           Very similar to other function, but we need to add gates 
           to nn.Parameter, during the forward pass'''     
        assert len(self.masks) <= self.n_contexts, "Contexts are more than the specified number" 
        if context not in self.learnable_parameters:
            "make the first unsigned key the new context"
            unsigned_key = [k for k in self.learnable_parameters.keys() if 'unsigned' in k][0]
            self.learnable_parameters[context] = self.learnable_parameters.pop(unsigned_key)
            self.masks[context] = self.masks.pop(unsigned_key[8:])
        if self.training:
            # make only the current context trainable
            if not self.activate_current_context:
                if context != self.current_context:
                    self.current_context = context
                self.set_current_grad_true(context)
        else:
            if not self.all_grads_false:
                self.set_grads_to_false()
        local_mask = self.masks[context] != 0
        full_mask = self.masks[context].clone().detach().requires_grad_(False)
        full_mask[local_mask] = self.learnable_parameters[context]
        return torch.tanh(full_mask) # learnable gates should be between -1, 1
        # return full_mask
    
    def set_current_grad_true(self, context):
        self.set_grads_to_false()
        self.learnable_parameters[context].requires_grad = True
        self.activate_current_context = True
        
    def get_transition_index(self, x):
        """Test python code for number of 1 in the branch dimension for a sparsity, replace 2 with n_b
        for i in range(10):
            print(f'{i/10}: {round((2)*(1-(i/10)))}')
        """
        assert self.sparsity >= 0 and self.sparsity <= 1, "Sparsity must be a value between 0 and 1"
        if self.sparsity == 0:
            return x
        if self.sparsity == 1:
            return 1
        else:
            potential_index = (x) * (1 - self.sparsity)
            transition_index = max(1, round(potential_index))
            return transition_index
    
    def generate_interpolated_array(self, x):
        """
        Generate a 1D tensor of size x, smoothly transitioning from 1's to 0's
        based on the input sparsity between 0 and 1.

        Parameters:
        - x: The size of the output tensor.
        - sparsity: A single sparsity between 0 and 1 determining the blend of 1's and 0's.

        Returns:
        - A Pyth tensor according to the specified rules.
        """
        # Calculate the transition index based on the sparsity
        transition_index = self.get_transition_index(x)

        # Create a tensor of 1's and 0's based on the transition index
        output = torch.zeros(x, dtype=torch.float32, device=self.device)
        output[:transition_index + 1] = 1
        #permute the tensor randomly
        output = output[torch.randperm(x)]
        return output    
        
    def make_learnable_gates(self, base):
        mask = base != 0
        values = nn.init.kaiming_normal_(torch.empty(mask.sum(),1, device=self.device)).squeeze(1).clone().detach().requires_grad_(True)
        learnable_gates = nn.Parameter(values).requires_grad_(True)
        return learnable_gates
        
    def set_grads_to_false(self):
        for k, param in self.learnable_parameters.items():
            param.requires_grad = False
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        self.all_grads_false = True
        self.activate_current_context = False
        
    def eval(self):
        self.set_grads_to_false()
        return super().eval()
    
    def __repr__(self):
        return super().__repr__() + f'\nBranchGatingActFunc(n_next_h={self.n_next_h}, \
    \nn_b={self.n_b}, n_contexts={self.n_contexts}, sparsity={self.sparsity},\
    \nlearn_gates={self.learn_gates}, soma_func={self.soma_act_func.__name__}, device={self.device},\
    \nforward={self.forward.__name__}'
        
        


def test_gating_act_func(branch=1):
    n_batches = 10
    n_b = branch
    n_next_h = 4
    n_contexts = 2
    for sparsity in [0, 0.5, 1]:
        gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, sparsity)
        x = torch.rand(n_batches, n_b, n_next_h)
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
        x = torch.rand(n_batches, n_next_h)
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
        x = torch.rand(n_batches, n_b, n_next_h)
        out = gating(x)
        assert out.shape == (n_batches, n_next_h), f'Expected shape {(n_batches, n_next_h)}, got {out.shape}'
    print("learnGates test passed")
    

def test_branch_gating_act_func():
    n_batches = 10
    n_b = 10
    n_next_h = 4
    n_contexts = 2
    sparsity_values = [0, 0.5, 1]
    
    for sparsity in sparsity_values:
        gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, sparsity)
        x = torch.rand(n_batches, n_b, n_next_h)
        out = gating(x)
        
        # Check output shape
        assert out.shape == (n_batches, n_next_h), f'Expected shape {(n_batches, n_next_h)}, got {out.shape}'
        
        # Verify parameter constraints
        assert 0 <= gating.sparsity <= 1, "Sparsity must be between 0 and 1"
        assert gating.n_b == n_b, "Branch count mismatch"
        assert gating.n_contexts == n_contexts, "Context count mismatch"
        assert gating.n_next_h == n_next_h, "Next hidden layer size mismatch"
        
    print("All tests passed for BranchGatingActFunc")
    
    
def test_gradient_backprop_multi_context(soma_func='sum',temp=1):
    torch.manual_seed(42)  # for reproducibility

    n_batches = 5
    n_b = 3
    n_next_h = 10
    n_contexts = 30  # Number of different contexts
    sparsity = 0.5
    learn_gates = True

    gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, 
                                 sparsity, learn_gates, 
                                 soma_func=soma_func)
    # print(f'gating soma_func: {gating.soma_act_func}')
    # Create a dummy optimizer, assuming learnable parameters are properly registered
    optimizer = torch.optim.SGD(gating.parameters(), lr=0.1)
    # set_trace()
    for context_index in [None] + [range(n_contexts)] :
        # Simulate switching context
        # gating.set_context(context_index)  # Assuming there is a method to switch context
        x = torch.randn(n_batches, n_b, n_next_h, requires_grad=True)
        target = torch.randn(n_batches, n_next_h)  # Random target for loss computation
        
        # Training step
        gating.train()  # Set the module to training mode
        optimizer.zero_grad()  # Clear previous gradients
        output = gating(x, context_index)
        loss = (output - target).pow(2).mean()  # Simple MSE loss
        loss.backward()
        optimizer.step()

        # Check if the gradients are correctly set for the current context
        for name, param in gating.named_parameters():
            if f"{context_index}" in name and 'unsigned' not in name:
                assert param.grad is not None and param.grad.abs().sum() > 0, f"Gradients should not be zero for current context {context_index}"
            else:
                assert param.grad is None or param.grad.abs().sum() == 0, f"Gradients should be zero for other contexts"

        # Evaluation step
        gating.eval()  # Set the module to evaluation mode
        with torch.no_grad():
            output = gating(x)
        
        # Check if no gradients accumulate during evaluation
        for name, param in gating.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, "No gradients should accumulate during evaluation"
        
        # Optionally verify some output property during evaluation
        # ...

    print("Gradient management test across multiple contexts passed.")
    
def test_soma_function_variation():
    # Define the test input
    test_input = torch.randn(10, 2, 784)  # Assuming the input dimensions need to be like this

    # Define device
    device = 'cpu'  # use 'cuda' if running on GPU

    # List of gate functions to test
    soma_functions = ['sum', 'median', 'max', 'softmax_2.0', 'softmaxsum_2.0']

    # Dictionary to store outputs
    outputs = {}

    # Create an instance of the model for each gate function and store the output
    for func in soma_functions:
        model = BranchGatingActFunc(n_next_h=784, n_b=2, n_contexts=1, sparsity=0.5, learn_gates=False, soma_func=func, device=device)
        # set_trace()
        model.to(device)
        test_input = test_input.to(device)
        
        # Assuming a single context (context 0) for simplicity
        output = model(test_input, context=0)
        
        # Store output
        outputs[func] = output

    keys = list(outputs.keys())
    num_keys = len(keys)
    similar_pairs = []

    # Compare every pair of outputs
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            output1 = outputs[keys[i]]
            output2 = outputs[keys[j]]

            # Check if outputs are close enough to be considered identical
            if torch.allclose(output1, output2, atol=1e-6):
                similar_pairs.append((keys[i], keys[j]))
    
    # Report findings
    if similar_pairs:
        for pair in similar_pairs:
            print(f"Outputs for gate functions '{pair[0]}' and '{pair[1]}' are identical.")
    else:
        print("No identical outputs found among the tested gate functions.")

    print("Test passed: All gate function outputs are different.")
    


class TestBranchGatingActFunc(unittest.TestCase):
    def test_masks_sparsity(self):
        # Test various combinations of branches and sparsity
        combinations = [(b, s / 10) for b in [1, 10, 20, 30, 100] for s in range(0, 11, 1)]
        n_next_h = 10  # Set a fixed number of neurons in the next hidden layer for testing

        for n_b, sparsity in combinations:
            with self.subTest(n_b=n_b, sparsity=sparsity):
                module = BranchGatingActFunc(n_next_h=n_next_h, n_b=n_b, sparsity=sparsity, device='cpu')
                mask = module.gen_branching_mask()

                # Determine the expected number of 1's per column based on sparsity
                if sparsity == 1.0:
                    expected_ones_per_column = 1
                elif sparsity == 0.0:
                    if n_b == 1:
                        expected_ones_per_column = n_next_h  # When n_b = 1 and sparsity is 0, all entries should be 1
                    else:
                        expected_ones_per_column = n_b  # All entries in each column should be 1 when sparsity is 0 and n_b > 1
                else:
                    if n_b == 1:
                        expected_ones_per_column = round(n_next_h * (1 - sparsity))  # Sparsity across the single row
                    else:
                        expected_ones_per_column = round(n_b * (1 - sparsity))  # Normal sparsity across columns

                # Check each column individually
                for col in range(mask.shape[1]):
                    actual_ones = torch.sum(mask[:, col]).item()
                    self.assertEqual(actual_ones, expected_ones_per_column,
                                     f"Column {col} expected {expected_ones_per_column} ones, got {actual_ones}\nMasks:\n{mask}")

    
if __name__ == "__main__":
    for b in [1,10, 20, 30 , 100]:
        test_gating_act_func()
    test_masse_act_func()
    test_learnable_gates()
    test_branch_gating_act_func()
    test_gradient_backprop_multi_context()
    for soma_func in ['sum', 'max', 'softmax_2.0', 'softmaxsum_2.0']:
        test_gradient_backprop_multi_context(soma_func)
        print(f'Gradient backprop test passed for {soma_func} gate function')
    test_soma_function_variation()
    unittest.main()
    