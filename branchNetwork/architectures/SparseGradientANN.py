import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Union
from ipdb import set_trace
import unittest
from torch.optim.optimizer import Optimizer, required


class AdamSI(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, c=1, xi=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, c=c, xi=xi)
        super(AdamSI, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    # Synaptic importance measures
                    state['omega'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running averages till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] * (beta2 ** state['step']).sqrt() / (1 - beta1 ** state['step'])

                # Apply regularization from SI
                if group['c'] > 0:
                    p.data.add_(-group['c'] * state['omega'], p.data - state['exp_avg'])

                # Update parameters
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Update omega for the next step
                state['omega'].add_(-group['xi'] * (p.data - state['exp_avg']).pow(2))

        return loss



class SparseGradientANN(nn.Module):
    def __init__(self, config):
        super(SparseGradientANN, self).__init__()
        trainable_percent = config['trainable_percent'] if 'trainable_percent' in config else 100
        self.trainable_percent = trainable_percent / 100.0  # Convert percentage to a proportion
        self.n_contexts = config.get('n_contexts', 5)
        
        input_size = config.get('input_size', 784) # config['input_size'] if 'input_size' in config else 784
        hidden_sizes = config.get('hidden_sizes', [784, 784]) # config['hidden_sizes'] if 'hidden_sizes' in config else [2000, 2000]
        output_size = config.get('output_size', 10) # config['output_size'] if 'output_size' in config else 10
        self.seen_contexts = []
        
        # Define layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )

        # Apply the masking to make some weights non-trainable
        self.current_context = 0
        self.hooks = {}
        self.apply_mask()
        self.previous_mode = self.training

    def forward(self, x, context=0):
        self.check_context(context)
        return self.layers(x)
    
    def check_context(self, context):
        if context not in self.seen_contexts:
            self.seen_contexts.append(context)
        new_index = self.seen_contexts.index(context)
        if self.current_context != new_index:
            self.current_context = new_index
            self.update_hooks()

            
    def prep_hook(self, name):
        def hook(grad):
            return grad * getattr(self, f'contxt_{getattr(self, "current_context")}_{name}_mask')
        return hook

    
    def apply_mask(self):
        # Loop over all parameters and apply mask
        for c in range(self.n_contexts):
            for name, param in self.named_parameters():
                if 'weight' in name:  # Apply only to weights, not biases
                    sanitized_name = name.replace('.', '_')  # Replace dots with underscores
                    total_weights = param.numel()  # Total number of weights in the layer
                    num_trainable = int(total_weights * self.trainable_percent)  # Number of trainable weights based on the percent
                    
                    # Create a mask with the specified number of trainable weights
                    mask = torch.zeros(total_weights, dtype=torch.bool)
                    perm = torch.randperm(total_weights)
                    mask[perm[:num_trainable]] = True  # Randomly select weights to be trainable
                    
                    # Reshape mask to the shape of the parameter matrix
                    mask = mask.reshape(param.shape)
                    
                    # Register buffer is used here to store the mask to not update gradients
                    self.register_buffer(f'contxt_{c}_{sanitized_name}_mask', mask)
                    
                    # Modify the gradient function of the parameter to apply mask
                    param.register_hook(self.prep_hook(sanitized_name))
                    
    def update_hooks(self):
        # Remove existing hooks
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

        # Apply new hooks for the current context
        for name, param in self.named_parameters():
            if 'weight' in name:
                sanitized_name = name.replace('.', '_')
                handle = param.register_hook(self.prep_hook(sanitized_name))
                self.hooks[sanitized_name] = handle
                
    def train(self, mode=True):
        super().train(mode)
        self.check_and_clear_gradients()
        
    def eval(self):
        super().eval()
        self.check_and_clear_gradients()
        
    def check_and_clear_gradients(self):
        if self.previous_mode != self.training:
            self.zero_grad()
        self.previous_mode = self.training
    
    
    
def test_multi_contexts(model_class):
    # Configuration for the BranchModel
    model_configs = {
        'n_in': 784, 'n_out': 10, 'n_npb': [2000, 2000], 'n_branches': [200, 200],
        'n_contexts': 5, 'sparsity': 0.1, 'learn_gates': True, 'soma_func': 'sum',
        'temp': 1.0, 'dropout': 0.5, 'device': 'cpu', 'trainable_percent': 20,
        'n_contexts':3, # Set device to 'cpu' for simplicity
    }
    
    # Instantiate the model with 50% of the weights trainable
    model = model_class(model_configs)
    for c in range(model.n_contexts):
        model.train()

        # Create dummy input and output
        dummy_input = torch.randn(1, 784)  # Example input tensor
        dummy_target = torch.tensor([2])   # Example target tensor, assuming a classification task with 10 classes
        criterion = torch.nn.CrossEntropyLoss()

        # Forward pass
        output = model(dummy_input, context=c)
        loss = criterion(output, torch.nn.functional.one_hot(dummy_target, num_classes=10).float())

        # Backward pass
        # set_trace()
        loss.backward()

        # Check if the weights that should not be trainable have remained unchanged
        all_weights_frozen = True
        for name, param in model.named_parameters():
            if hasattr(param, 'trainable_mask'):
                # Mask should be present and applied to weights
                untrainable_weights = param.data * (param.trainable_mask == 0)
                untrainable_weights_initial = untrainable_weights.detach().clone()

                # After updating weights, check if untrainable weights have changed
                if not torch.allclose(untrainable_weights, untrainable_weights_initial):
                    all_weights_frozen = False
                    print(f"Test failed: Weights in {name} were unexpectedly updated.")
                    break

        if all_weights_frozen:
            print("Test passed: Non-trainable weights remain unchanged.")
        else:
            print("Test failed: Some non-trainable weights were updated.")
          
          
def test_gradient_hooks():
    # Configuration for a simple test
    config = {
        'trainable_percent': 50,  # 50% of weights are trainable
        'n_contexts': 3,
        'input_size': 4,
        'hidden_sizes': [10, 10],
        'output_size': 3,
        'device': 'cpu'
    }

    # Initialize the model
    model = SparseGradientANN(config)
    
    # Create dummy data to pass through the model
    input_tensor = torch.randn(10, config['input_size'], requires_grad=True)
    target = torch.randint(0, config['output_size'], (10,))

    # Train under different contexts
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for context in range(config['n_contexts']):
        model.train()
        optimizer.zero_grad()
        
        output = model(input_tensor, context=context)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Check gradients
        print(f"Context {context}:")
        for name, param in model.named_parameters():
            if 'weight' in name:
                sanitized_name = name.replace('.', '_')
                mask = getattr(model, f'contxt_{context}_{sanitized_name}_mask')
                masked_grad = param.grad * mask.float()
                
                # Ensure the unmasked gradients are non-zero
                if torch.any(masked_grad != param.grad):
                    print(f"  {name} has masked gradients.")
                else:
                    print(f"  {name} has no masked gradients.")
        
        # Verify no old hooks are active
        model.eval()
        optimizer.zero_grad()
        output = model(input_tensor, context=context)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Now verify that the gradients are still correctly masked (i.e., no leakage from old contexts)
        all_gradients_correct = True
        for name, param in model.named_parameters():
            if 'weight' in name:
                sanitized_name = name.replace('.', '_')
                mask = getattr(model, f'contxt_{context}_{sanitized_name}_mask')
                masked_grad = param.grad * mask.float()
                if not torch.allclose(masked_grad, param.grad):
                    all_gradients_correct = False
                    print(f"Error: {name} in context {context} has incorrect gradients after re-checking.")
        
        if all_gradients_correct:
            print(f"  Gradients in context {context} are correct after re-checking.")


class TestModeSwitch(unittest.TestCase):
    def test_gradient_clearance(self):
        # Create an instance of the network
        net = SparseGradientANN({})
        # Create a dummy input
        input_tensor = torch.randn(10, 784, requires_grad=True)
        # Run a forward pass
        output = net(input_tensor)
        # Run a backward pass to generate some gradients
        output.sum().backward()

        # Check if gradients exist before mode switch
        for param in net.parameters():
            self.assertIsNotNone(param.grad)
            print("Gradients exist before switching modes.")

        # Switch to eval mode and check gradients
        net.eval()
        for param in net.parameters():
            self.assertIsNone(param.grad)  # Gradients should be cleared
            print("Gradients cleared after switching to eval.")

        # Switch back to train mode and check again
        net.train()
        for param in net.parameters():
            self.assertIsNone(param.grad)  # Gradients should remain cleared
            print("Gradients remain cleared after switching back to train.")

         
if __name__ == '__main__':
    test_multi_contexts(SparseGradientANN)
    # Run the test
    test_gradient_hooks()
    unittest.main()