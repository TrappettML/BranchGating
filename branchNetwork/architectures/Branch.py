import torch
from torch import nn

from branchNetwork.BranchLayer import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from typing import Union


class BranchModel(nn.Module):
        '''We want the same number of weights for each layer as Masse has.
        layer 1 is 784x2000, layer2 is 2000x2000, layer3 is 2000x10'''
        def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
            super(BranchModel, self).__init__()
            # self.layer_1 = nn.Linear(model_configs['n_in'], 2000)
            self.layer_1 = BranchLayer(model_configs['n_in'],
                                      model_configs['n_npb'][0],
                                       model_configs['n_branches'][0],
                                       2000,
                                       device=model_configs['device'])
            self.gating_1 = BranchGatingActFunc(2000,
                                                model_configs['n_branches'][0],
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'])
            self.layer_2 = BranchLayer(2000,
                                      model_configs['n_npb'][1],
                                       model_configs['n_branches'][1],
                                       2000,
                                       device=model_configs['device'])
            self.gating_2 = BranchGatingActFunc(2000,
                                                model_configs['n_branches'][1],
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'])
            
            self.layer_3 = nn.Linear(2000, model_configs['n_out'])
            self.drop_out = nn.Dropout(model_configs['dropout'])
            self.act_func = nn.ReLU()  
                     
        def forward(self, x, context=0):
            x = self.drop_out(self.act_func(self.gating_1(self.layer_1(x), context)))
            x = self.drop_out(self.act_func(self.gating_2(self.layer_2(x), context)))
            return self.layer_3(x)
        
        
def test_Branch():
    model_configs = {'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': 5, 
                    'device': 'cpu', 
                    'n_npb': [56, 200], 
                    'n_branches': [14, 10], 
                    'sparsity': 0.8,
                    'dropout': 0,}
    
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