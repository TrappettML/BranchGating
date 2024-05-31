import torch
from torch import nn

from branchNetwork.BranchLayerMM import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from typing import Union


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
                                                learn_gates)
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
                                                temp=temp)
            
            self.layer_3 = nn.Linear(784, model_configs['n_out'])
            self.drop_out = nn.Dropout(model_configs['dropout'])
            self.act_func = nn.ReLU()  
                     
        def forward(self, x, context=0):
            x = self.drop_out(self.act_func(self.gating_1(self.layer_1(x), context)))
            x = self.drop_out(self.act_func(self.gating_2(self.layer_2(x), context)))
            return self.layer_3(x)
        
        
def test_Branch(gate_func='sum', temp=1):
    model_configs = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': 5, 
                    'device': 'cpu', 
                    'n_npb': [56, 56], 
                    'n_branches': [14, 14], 
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
        
if __name__ == '__main__':
    test_Branch()
    print('BranchModel test passed.')
    for gate_func in ['sum', 'softmax', 'max']:
        for temp in [0.1, 1, 10]:
            test_Branch(gate_func, temp)
            print(f'BranchModel test passed for gate_func: {gate_func} and temp: {temp}')