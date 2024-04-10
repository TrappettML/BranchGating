
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(SimpleMLP, self).__init__()
        layers = []

        # Input layer
        prev_layer_size = input_size

        # Add hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, hidden_dim))
            layers.append(nn.ReLU())  # Using ReLU activation function for hidden layers
            prev_layer_size = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_layer_size, output_size))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, context=0):
        return self.model(x)