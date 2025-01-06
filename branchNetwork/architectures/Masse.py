import torch
from torch import nn

from branchNetwork.BranchLayerMM import BranchLayer
from branchNetwork.gatingActFunction import BranchGatingActFunc
from typing import Union
from ipdb import set_trace


class MasseModel(nn.Module):
    def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
        super(MasseModel, self).__init__()
        learn_gates = model_configs['learn_gates'] if 'learn_gates' in model_configs else False
        self.layer_1 = nn.Linear(model_configs['n_in'], 784)
        self.layer_2 = nn.Linear(784, 784)
        self.layer_3 = nn.Linear(784, model_configs['n_out'])
        det_gates = model_configs.get('det_masks', False)
        sparsities = model_configs.get('sparsity', 0)
        if type(sparsities) == tuple:
            sparsity_1 = sparsities[0]
            sparsity_2 = sparsities[1]
        else:
            sparsity_1 = sparsities
            sparsity_2 = sparsities
        self.gating_1 = BranchGatingActFunc(784,
                                            1,
                                            model_configs['n_contexts'],
                                            sparsity_1,
                                            learn_gates,
                                            det_masks=det_gates)
        self.gating_2 = BranchGatingActFunc(784,
                                            1,
                                            model_configs['n_contexts'],
                                            sparsity_2,
                                            learn_gates,
                                            det_masks=det_gates)
        self.act_func = nn.ReLU()  
        self.drop_out = nn.Dropout(model_configs.get('dropout', 0)) 
                
    def forward(self, x, context=0):
        print(f'Inpute shape: {x.shape}')
        set_trace()
        x = self.drop_out(self.act_func(self.gating_1(self.layer_1(x), context)))
        print(f'after l1: {x.shape}')
        x = self.drop_out(self.act_func(self.gating_2(self.layer_2(x), context)))
        print(f'after l2: {x.shape}')
        return self.layer_3(x)
