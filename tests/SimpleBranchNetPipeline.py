# __init__.py


import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm
from ipdb import set_trace

from branchNetwork.simplebranchlayer import BranchLayer
from branchNetwork.simpleMLP import SimpleMLP


def load_mnist_data(batch_size=32, download=True, root='./data'):
    """
    Load the MNIST dataset.

    Parameters:
    - batch_size: The number of samples per batch to load.
    - download: Whether to download the dataset if it's not already available locally.
    - root: The directory where the dataset will be stored.

    Returns:
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the test data.
    """
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
        transforms.Lambda(lambda x: torch.flatten(x)), # Flatten the images
    ])

    # Load the training and test datasets
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    # Create data loaders for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class BranchFirstNet(nn.Module):
    def __init__(self, hidden_layers, output_size, branch_params):
        super(BranchFirstNet, self).__init__()
        self.branch = BranchLayer(branch_params)
        self.branch_out_size = self.branch._output_size()
        self.mlp = SimpleMLP(self.branch_out_size, hidden_layers, output_size)
        
    def forward(self, x):
        x = self.branch(x)
        out = self.mlp(x)
        return out


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
   

def evaluate_model(model, eval_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Optionally, calculate the number of correct predictions
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct / len(eval_loader.dataset) * 100
    return avg_loss
    
        
# Plot the training and test losses
def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./branch_first_net_losses.png')
    
def main():
    # Training settings
    batch_size = 64
    epochs = 100
    lr = 0.01
    momentum = 0.5
    seed = 1
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_loader, test_loader = load_mnist_data(batch_size, download=True, root='./data')

    # Create the model
    branch_params = {'n_inputs': 784, 'branching_factor': 10, 'device': device}
    model = BranchFirstNet([16], 10, branch_params).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    all_train_loss = []
    all_test_loss = []
    for epoch in tqdm(range(1, epochs + 1)):
        all_train_loss.append(train(model, device, train_loader, optimizer, criterion, epoch))
        all_test_loss.append(evaluate_model(model, test_loader, criterion, device))
        
    plot_losses(all_train_loss, all_test_loss)
        
        
if __name__ == '__main__':
    main()