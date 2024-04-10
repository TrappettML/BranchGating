import torch.nn as nn
import torch
from branchNetwork.architectures.Simple import SimpleModel
from typing import Union

class ExpertModel(nn.Module):
        def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
            super(ExpertModel, self).__init__()
            self.seen_contexts = list()
            self.n_contexts = model_configs['n_contexts']
            self.models = [SimpleModel(model_configs) for _ in range(model_configs['n_contexts'])]
            self.current_model = self.models[0]
            
        def forward(self, x, context=0):
            self.check_context(context)
            return self.current_model(x)
        
        def check_context(self, context):
            '''check if it is a new context, if it is, switch to that model'''
            if context not in self.seen_contexts:
                self.seen_contexts.append(context)
                assert len(self.seen_contexts) <= self.n_contexts, "Contexts are more than the specified number" 
                self.current_model = self.models[self.seen_contexts.index(context)]
                

    