# __init__.py
import torch as th
from torch import nn
from ipdb import set_trace

class BranchGatingActFunc(nn.Module):
    def __init__(self, n_next_h, n_b=1, n_contexts=1, sparsity=0, learn_gates=False, gate_func='sum',temp=1, device='cpu'):
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
        self.gate_act_func = self.set_gate_func(gate_func, temp)
        self.device = device
        if learn_gates:
            self.make_learnable_parameters()
            self.all_grads_false = False
            self.activate_current_context = False
    
    def set_gate_func(self, gate_func, temp):
        if gate_func == 'sum':
            return th.sum
        elif gate_func == 'median':
            def my_median(x):
                return th.median(x, dim=1)[0]
            return my_median
        elif gate_func == 'max':
            def my_max(x):
                return th.max(x, dim=1)[0]
            return my_max
        elif gate_func == 'softmax':
            def my_softmax(x):
                sm = nn.functional.softmax(x/temp, dim=1)
                sm_2d = sm.view(-1, sm.size(1))
                sampled_indices = th.multinomial(sm_2d, num_samples=1)
                sampled_indices = sampled_indices.view(x.size(0), x.size(2)) 
                batch_indices = th.arange(x.size(0)).unsqueeze(-1).expand_as(sampled_indices)
                max_x = x[batch_indices, batch_indices, th.arange(x.size(2))]
                return max_x
            return my_softmax
        elif gate_func == 'softmax_sum':
            def my_smx_sum(x):
                sm = nn.functional.softmax(x/temp, dim=1)
                sm_mm = x * sm
                return th.sum(sm_mm, dim=1)
            return my_smx_sum
        else:
            raise ValueError(f"gate_func must be one of ['sum', 'max', 'softmax', 'softmax_sum'], got {gate_func}")
        
    def make_learnable_parameters(self):
        self.learnable_parameters = nn.ParameterDict()
        for i in range(self.n_contexts):
            self.masks[str(i)] = self.make_mask()
            self.learnable_parameters['unsigned' + str(i)] = self.make_learnable_gates(self.masks[str(i)])
        
    def make_mask(self):
        if self.n_b == 1: # will be Masse style Model
            mask = self.generate_interpolated_array(self.n_next_h, self.sparsity)
            return mask.float()
        else:
            return self.gen_branching_mask()
        
    def _gen_branching_mask(self):
        return th.stack([self.generate_interpolated_array(self.n_b, self.sparsity) for _ in range(self.n_next_h)]).float().T
    
    def gen_branching_mask(self):
        t_i = self.get_transition_index()
        mask = th.zeros(self.n_b, self.n_next_h, dtype=th.float32, device=self.device)
        mask[:t_i, :] = 1
        random_indices = th.argsort(th.rand(self.n_b, self.n_next_h, device=self.device), dim=0)
        mask = th.gather(mask, 0, random_indices)
        return mask
        
    def branch_forward(self, x, context=0):
        x = x.to(self.device)
        '''forward function for when n_b > 1
           sum over the n_b dimension'''
        context = str(context)
        gate = self.get_context(context)
        return th.sum(x * gate, dim=1) # x is shape (n_batches, n_b, n_next_h), and so we are summing over branches. 
    
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
            self.masks[context] = self.make_mask()
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
        return th.tanh(full_mask)
        # return full_mask
    
    def set_current_grad_true(self, context):
        self.set_grads_to_false()
        self.learnable_parameters[context].requires_grad = True
        self.activate_current_context = True
        
    def get_transition_index(self):
        assert self.sparsity >= 0 and self.sparsity <= 1, "Sparsity must be a value between 0 and 1"
        transition_index = round((self.n_b - 1) * (1 - self.sparsity))
        return transition_index
    
    def generate_interpolated_array(self, x, sparsity):
        """
        Generate a 1D tensor of size x, smoothly transitioning from 1's to 0's
        based on the input sparsity between 0 and 1.

        Parameters:
        - x: The size of the output tensor.
        - sparsity: A single sparsity between 0 and 1 determining the blend of 1's and 0's.

        Returns:
        - A Pyth tensor according to the specified rules.
        """
        assert x > 0, "x must be greater than 0"
        assert sparsity >= 0, "sparsity must be greater than or equal to 0"
        assert sparsity <= 1, "sparsity must be less than or equal to 1"

        # Calculate the transition index based on the sparsity
        transition_index = round((x - 1) * (1 - sparsity))

        # Create a tensor of 1's and 0's based on the transition index
        output = th.zeros(x, dtype=th.float32, device=self.device)
        output[:transition_index + 1] = 1
        #permute the tensor randomly
        output = output[th.randperm(x)]
        return output    
        
    def make_learnable_gates(self, base):
        mask = base != 0
        values = nn.init.kaiming_normal_(th.empty(mask.sum(),1, device=self.device)).squeeze(1).clone().detach().requires_grad_(True)
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
        
        


def test_gating_act_func(branch=1):
    n_batches = 10
    n_b = branch
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
    

def test_branch_gating_act_func():
    n_batches = 10
    n_b = 10
    n_next_h = 4
    n_contexts = 2
    sparsity_values = [0, 0.5, 1]
    
    for sparsity in sparsity_values:
        gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, sparsity)
        x = th.rand(n_batches, n_b, n_next_h)
        out = gating(x)
        
        # Check output shape
        assert out.shape == (n_batches, n_next_h), f'Expected shape {(n_batches, n_next_h)}, got {out.shape}'
        
        # Verify parameter constraints
        assert 0 <= gating.sparsity <= 1, "Sparsity must be between 0 and 1"
        assert gating.n_b == n_b, "Branch count mismatch"
        assert gating.n_contexts == n_contexts, "Context count mismatch"
        assert gating.n_next_h == n_next_h, "Next hidden layer size mismatch"
        
    print("All tests passed for BranchGatingActFunc")
    
    
def test_gradient_backprop_multi_context(gate_func='sum',temp=1):
    th.manual_seed(42)  # for reproducibility

    n_batches = 5
    n_b = 3
    n_next_h = 10
    n_contexts = 30  # Number of different contexts
    sparsity = 0.5
    learn_gates = True

    gating = BranchGatingActFunc(n_next_h, n_b, n_contexts, 
                                 sparsity, learn_gates, 
                                 gate_func=gate_func,temp=temp)
    # print(f'gating gate_func: {gating.gate_act_func}')
    # Create a dummy optimizer, assuming learnable parameters are properly registered
    optimizer = th.optim.SGD(gating.parameters(), lr=0.1)
    
    for context_index in [None] + [range(n_contexts)] :
        # Simulate switching context
        # gating.set_context(context_index)  # Assuming there is a method to switch context
        x = th.randn(n_batches, n_b, n_next_h, requires_grad=True)
        target = th.randn(n_batches, n_next_h)  # Random target for loss computation
        
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
        with th.no_grad():
            output = gating(x)
        
        # Check if no gradients accumulate during evaluation
        for name, param in gating.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, "No gradients should accumulate during evaluation"
        
        # Optionally verify some output property during evaluation
        # ...

    print("Gradient management test across multiple contexts passed.")
    
if __name__ == "__main__":
    for b in [1,10, 20, 30 , 100]:
        test_gating_act_func()
    test_masse_act_func()
    test_learnable_gates()
    test_branch_gating_act_func()
    test_gradient_backprop_multi_context()
    for gate_func in ['sum', 'max', 'softmax', 'softmax_sum']:
        test_gradient_backprop_multi_context(gate_func)
        print(f'Gradient backprop test passed for {gate_func} gate function')