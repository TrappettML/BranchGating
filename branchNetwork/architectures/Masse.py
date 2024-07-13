import torch
from torch import nn

from branchNetwork.BranchLayerMM import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from typing import Union


class MasseModel(nn.Module):
        def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
            super(MasseModel, self).__init__()
            learn_gates = model_configs['learn_gates'] if 'learn_gates' in model_configs else False
            self.layer_1 = nn.Linear(model_configs['n_in'], 784)
            self.layer_2 = nn.Linear(784, 784)
            self.layer_3 = nn.Linear(784, model_configs['n_out'])
            self.gating_1 = BranchGatingActFunc(784,
                                                1,
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'],
                                                learn_gates)
            self.gating_2 = BranchGatingActFunc(784,
                                                1,
                                                model_configs['n_contexts'],
                                                model_configs['sparsity'],
                                                learn_gates)
            self.act_func = nn.ReLU()  
            self.drop_out = nn.Dropout(model_configs['dropout']) 
                    
        def forward(self, x, context=0):
            x = self.drop_out(self.act_func(self.gating_1(self.layer_1(x), context)))
            x = self.drop_out(self.act_func(self.gating_2(self.layer_2(x), context)))
            return self.layer_3(x)
        
        
