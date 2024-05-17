import torch.nn as nn
import torch
from typing import Union

class SimpleModel(nn.Module):
    def __init__(self, model_configs: dict[str, Union[str, int, float, dict]]):
        super(SimpleModel, self).__init__()
        layers = [nn.Linear(model_configs['n_in'], 784),
                  nn.ReLU(),
                  nn.Dropout(model_configs['dropout']),
                  nn.Linear(784, 784),
                  nn.ReLU(),
                  nn.Dropout(model_configs['dropout']),
                  nn.Linear(784, model_configs['n_out'])]

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, context=0):
        return self.model(x)
    
    
    
