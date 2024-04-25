import torch.nn as nn
import torch
from branchNetwork.architectures.Simple import SimpleModel
from typing import Union

class ExpertModel(nn.Module):
        def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
            super(ExpertModel, self).__init__()
            self.seen_contexts = list()
            self.n_contexts = model_configs['n_contexts']
            self.models = nn.ModuleList([SimpleModel(model_configs) for _ in range(model_configs['n_contexts'])])
            self.current_model_index = 0
            self.all_grads_false = False
            self.activate_training()
            
        def forward(self, x, context=0):
            if self.training:
                return self.train_forward(x, context)
            else:
                if not self.all_grads_false:
                    self.set_grads_to_false()
                return self.eval_forward(x, context)
            
        
        def check_context(self, context):
            '''check if it is a new context, if it is, switch to that model'''
            if context not in self.seen_contexts:
                self.seen_contexts.append(context)
                assert len(self.seen_contexts) <= self.n_contexts, "Contexts are more than the specified number" 
            

        def set_index(self, context):
            if self.current_model_index != self.seen_contexts.index(context):
                self.current_model_index = self.seen_contexts.index(context)
                self.activate_training()
                
        def activate_training(self):
            '''activate training for the current model'''
            for param in self.models[self.current_model_index].parameters():
                param.requires_grad = True
            for i, model in enumerate(self.models):
                if i != self.current_model_index:
                    for param in model.parameters():
                        param.requires_grad = False      
            self.all_grads_false = False
                        
        def train_forward(self, x, context):
            '''train forward pass'''
            self.check_context(context)
            self.set_index(context)
            if self.all_grads_false:
                self.activate_training()
            return self.models[self.current_model_index](x)
        
        def eval_forward(self, x, context):
            self.check_context(context)
            return self.models[self.seen_contexts.index(context)](x)
        
        def set_grads_to_false(self):
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
            self.all_grads_false = True
                

    