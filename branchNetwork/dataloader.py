
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import socket
from ipdb import set_trace


if 'talapas' in socket.gethostname():
    DATA_DIR = '/home/mtrappet/BranchGating/branchNetwork/data/'
else:
    DATA_DIR = '/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/'


def load_mnist_data(batch_size=32, download=True):
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
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor images
    ])

    # Load the training and test datasets
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=download, transform=transform)

    # Create data loaders for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_emnist():
    split = 'mnist'
    def get_emnist(root, train, download, transform, split=split):
        return datasets.EMNIST(root=root, split=split, download=download, transform=transform, train=train)
    return get_emnist

def load_rotated_flattened_data(dataset_name='MNIST', batch_size=32, rotation_in_degrees=0, download=True, root=DATA_DIR):
    """
    Load a dataset (e.g., MNIST, FashionMNIST, KMNIST) with each image rotated and flattened.

    Parameters:
    - dataset_name: Name of the dataset to load ('MNIST', 'FashionMNIST', 'KMNIST', etc.).
    - batch_size: The number of samples per batch to load.
    - rotation_in_degrees: The degree of rotation to apply to each image.
    - download: Whether to download the dataset if not locally available.
    - root: Directory where the dataset will be stored.

    Returns:
    - train_loader: DataLoader for the rotated and flattened training data.
    - test_loader: DataLoader for the rotated and flattened test data.
    """
    
    # Dataset dictionary to select from
    dataset_dict = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'KMNIST': datasets.KMNIST,
        'EMNIST': load_emnist(),
        # Add more datasets here if needed
    }
    
    if dataset_name not in dataset_dict:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from {list(dataset_dict.keys())}.")

    # Get the appropriate dataset class from the dictionary
    DatasetClass = dataset_dict[dataset_name]

    # Define the transformation: rotate, convert to tensor, normalize, and then flatten
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=[rotation_in_degrees, rotation_in_degrees]),  # Apply the specified rotation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
    ])

    # Load the training and test datasets with the specified transforms
    train_dataset = DatasetClass(root=root, train=True, download=download, transform=transform)
    test_dataset = DatasetClass(root=root, train=False, download=download, transform=transform)

    # Create data loaders for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_rotated_flattened_mnist_data(batch_size=32, rotation_in_degrees=0, download=True, root=DATA_DIR):
    """
    Load the MNIST dataset with each image rotated by a fraction of pi radians and flattened.

    Parameters:
    - batch_size: The number of samples per batch to load.
    - fraction_of_pi: The fraction of pi radians to rotate each image.
    - download: Whether to download the dataset if not locally available.
    - root: Directory where the dataset will be stored.

    Returns:
    - train_loader: DataLoader for the rotated and flattened training data.
    - test_loader: DataLoader for the rotated and flattened test data.
    """
    # Convert fraction of pi radians to degrees
    degrees = rotation_in_degrees

    # Define the transformation: rotate, convert to tensor, normalize, and then flatten
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=[degrees, degrees]),  # Apply the specified rotation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
    ])

    # Load the training and test datasets with the specified transforms
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=download, transform=transform)

    # Create data loaders for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def generate_permutation_indices(image_size, permutation_percent):
    """
    Generate a permutation index array for an image of size `image_size x image_size` 
    with the specified `permutation_percent` of pixels swapped.
    
    Parameters:
    - image_size: Size of one side of the image (e.g., 28 for a 28x28 image)
    - permutation_percent: Percentage of pixels to permute.
    
    Returns:
    - permuted_indices: A 1D array of permuted indices that can be applied to any image.
    """
    num_pixels = image_size * image_size
    indices = list(range(num_pixels))

    # Calculate the number of pixels to swap
    num_swaps = int((permutation_percent / 100.0) * (num_pixels // 2))

    # Randomly shuffle indices
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)

    # Swap the pixels according to the calculated number of swaps
    for i in range(num_swaps):
        idx1, idx2 = shuffled_indices[2 * i], shuffled_indices[2 * i + 1]
        indices[idx1], indices[idx2] = indices[idx2], indices[idx1]
    
    return indices


def load_permuted_flattened_mnist_data(batch_size=32, permutation_percent=0, download=True, root=DATA_DIR):
    """
    Load the MNIST dataset with each image having a percentage of its pixels permuted and flattened.

    Parameters:
    - batch_size: The number of samples per batch to load.
    - permutation_percent: The percentage of pixels to permute, from 0% (no permutation) to 100% (maximum permutation).
    - download: Whether to download the dataset if not locally available.
    - root: Directory where the dataset will be stored.

    Returns:
    - train_loader: DataLoader for the permuted and flattened training data.
    - test_loader: DataLoader for the permuted and flattened test data.
    """

    # Define the transformation: convert to tensor, permute pixels, normalize, and then flatten
    new_indices = generate_permutation_indices(28, permutation_percent)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
        transforms.Lambda(lambda x: torch.flatten(x)),
        transforms.Lambda(lambda x: x[new_indices]),  # Apply pixel permutation
        
    ])

    # Load the training and test datasets with the specified transforms
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    # Create data loaders for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    

def show_and_save_permuted_mnist_examples(permutation_percents):
    import matplotlib.pyplot as plt

    """
    Load MNIST data, find one example of each digit, show and save the figure,
    with each row showing the digits permuted by the corresponding percentage in permutation_percents.
    """
    # Find one example of each digit
    def find_example_images(data_loader):
        digit_images = [None] * 10
        found_digits = set()
        for images, labels in data_loader:
            for i, label in enumerate(labels):
                if label.item() in found_digits:
                    continue
                digit_images[label.item()] = images[i].view(28, 28).cpu().numpy()  # Reshape to 28x28 and store as numpy
                found_digits.add(label.item())
                if len(found_digits) == 10:
                    break
            if len(found_digits) == 10:
                break
        return digit_images

    # Plot the images with pixel permutations
    num_permutations = len(permutation_percents)
    fig, axes = plt.subplots(num_permutations, 10, figsize=(15, 1.5 * num_permutations))
    
    for i, percent in enumerate(permutation_percents):
        # Load the dataset with the current permutation percent
        train_loader, _ = load_permuted_flattened_mnist_data(batch_size=1000, permutation_percent=percent, root=DATA_DIR)

        # Get one example of each digit after permutation
        digit_images = find_example_images(train_loader)

        # Plot the permuted images
        for j, img in enumerate(digit_images):
            ax = axes[i, j] if num_permutations > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            if i == 0:
                ax.set_title(str(j))
            ax.axis('off')

    plt.savefig(f'./permuted_mnist_examples_via_loader.png')


def show_and_save_rotated_mnist_examples(dataset='MNIST'):
    import matplotlib.pyplot as plt

    """
    Load MNIST data, find one example of each digit, show and save the figure,
    with each row showing the digits permuted by the corresponding percentage in permutation_percents.
    """
    # Find one example of each digit
    def find_example_images(data_loader):
        digit_images = [None] * 10
        found_digits = set()
        for images, labels in data_loader:
            for i, label in enumerate(labels):
                if label.item() in found_digits:
                    continue
                digit_images[label.item()] = images[i].view(28, 28).cpu().numpy()  # Reshape to 28x28 and store as numpy
                found_digits.add(label.item())
                if len(found_digits) == 10:
                    break
            if len(found_digits) == 10:
                break
        return digit_images

    # Plot the images with pixel permutations
    rotations = list(range(0, 360, 36))
    fig, axes = plt.subplots(len(rotations), 10, figsize=(15, 1.5 * len(rotations)))
    
    for i, rotation in enumerate(rotations):
        # Load the dataset with the current permutation percent
        train_loader, _ = load_rotated_flattened_data(dataset, rotation_in_degrees=rotation, root=DATA_DIR)

        # Get one example of each digit after permutation
        digit_images = find_example_images(train_loader)

        # Plot the permuted images
        for j, img in enumerate(digit_images):
            ax = axes[i, j] if len(rotations) > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            if i == 0:
                ax.set_title(str(j))
            # ax.axis('off')
            axes[i,j].tick_params(axis='both', which='both',length=0)
            plt.setp(axes[i,j].get_xticklabels(), visible=False)
            plt.setp(axes[i,j].get_yticklabels(), visible=False)
        # Add the rotation label on the left side of each row
        axes[i, 0].set_ylabel(f'{rotation}°')
        
        # set_trace()
        
    plt.savefig(f'./rotated_{dataset}_examples_via_loader.png')


if __name__=='__main__':
    # show_and_save_permuted_mnist_examples([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    show_and_save_rotated_mnist_examples('EMNIST')
