# __init__.py


import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm
from ipdb import set_trace

from branchNetwork.BranchLayer import BranchLayer
from branchNetwork.simpleMLP import SimpleMLP

import ray

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

class BranchSecondNet(nn.Module):
    def __init__(self, hidden_layers, output_size, branch_params):
        super(BranchSecondNet, self).__init__()
        print(f'Branch params: {branch_params}')
        self.branch = BranchLayer(**branch_params)
        self.mlp = SimpleMLP(784, hidden_layers, output_size)
        
    def forward(self, x):
        # set_trace()
        x = self.mlp(x)
        x = self.branch(x)
        x = x.squeeze(1)
        return x


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
def plot_losses(train_losses, test_losses, file_name):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{file_name}.png')
    
def plot_branch_simple_losses(branch_train_losses, branch_test_loss, simple_train_losses, simple_test_losses, file_name):
    plt.plot(branch_train_losses, label='Branch Training Loss')
    plt.plot(branch_test_loss, label='Branch Test Loss')
    plt.plot(simple_train_losses, label='Simple Model Training Loss')
    plt.plot(simple_test_losses, label='Simple Model Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{file_name}.png')
    
@ray.remote(num_cpus=3, num_gpus=0)
def train_branching_model(train_loader, test_loader, device, lr=0.01, momentum=0.5, epochs=30):
    device = torch.device("cuda" if ray.get_gpu_ids() else "cpu") 
    branch_params = {
        'n_in': 64,
        'n_npb': 10,
        'n_b': 1,
        'n_next_h': 10,
    }
    model = BranchSecondNet([16], 64, branch_params).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    all_branch_train_loss = []
    all_branch_test_loss = []
    for epoch in range(1, epochs + 1):
        all_branch_train_loss.append(train(model, device, train_loader, optimizer, criterion, epoch))
        all_branch_test_loss.append(evaluate_model(model, test_loader, criterion, device))
    print(f'Branching model training complete')
    return all_branch_train_loss, all_branch_test_loss

@ray.remote(num_cpus=3, num_gpus=0)
def train_simple_model(train_loader, test_loader, device, lr=0.01, momentum=0.5, epochs=30):
    device = torch.device("cuda" if ray.get_gpu_ids() else "cpu") 
    simple_model = SimpleMLP(784, [16, 16], 10).to(device)
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    all_simple_train_loss = []
    all_simple_test_loss = []
    for epoch in range(1, epochs + 1):
        all_simple_train_loss.append(train(simple_model, device, train_loader, optimizer, criterion, epoch))
        all_simple_test_loss.append(evaluate_model(simple_model, test_loader, criterion, device))
    print(f'Simple model training complete')
    return all_simple_train_loss, all_simple_test_loss
    
def main():
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the data
    train_loader, test_loader = load_mnist_data(batch_size, download=True, root='./data')
    ray.init(num_cpus=6, num_gpus=0)
    # Wrap functions for Ray execution
    results = ray.get([train_branching_model.remote(train_loader, test_loader, device), train_simple_model.remote(train_loader, test_loader, device)])

    # Retrieve results
    branch_train_loss, branch_test_loss = results[0]
    simple_train_loss, simple_test_loss = results[1]
    ray.shutdown()
    # plot_losses(all_simple_train_loss, all_simple_test_loss, 'simple_mlp_losses')
    plot_branch_simple_losses(branch_train_loss, branch_test_loss, simple_train_loss, simple_test_loss, 'branch_simple_losses')
    print("All done!")
        
if __name__ == '__main__':
    main()