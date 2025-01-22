
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import socket
from ipdb import set_trace
from typing import List, Optional, Iterator
import random



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


def compute_similarity(perm: List[int], identity: List[int]) -> int:
    """
    Compute similarity score between a permutation and the identity permutation.
    
    Similarity is defined as the number of elements that are in the same position.
    
    Parameters:
    - perm (List[int]): Permuted list.
    - identity (List[int]): Identity list.
    
    Returns:
    - int: Number of elements in the same position.
    """
    return sum([1 for p, i in zip(perm, identity) if p == i])


class PermutedLabelMapper:
    def __init__(self, label_list: List[int], num_permutations: int = 100):
        """
        Initialize the PermutedLabelMapper.
        
        Parameters:
        - label_list (List[int]): The list of unique labels to permute.
        - num_permutations (int): Number of permutations to generate (including identity).
        """
        if num_permutations < 1:
            raise ValueError("num_permutations must be at least 1")
        
        self.label_list = label_list.copy()
        self.num_permutations = num_permutations
        self.identity = label_list.copy()
        self.permutations = self.generate_permutations()
    
    def generate_permutations(self) -> List[List[int]]:
        """
        Generate a list of permutations sorted by similarity to identity.
        
        Parameters:
        - None
        
        Returns:
        - List[List[int]]: List of permutations ordered from most similar to least similar.
        """
        # Start with identity
        permutations = [self.identity.copy()]
        seen = set()
        seen.add(tuple(self.identity))
        
        # Number of permutations to generate excluding identity
        num_to_generate = self.num_permutations - 1
        
        attempts = 0
        max_attempts = num_to_generate * 10  # Prevent infinite loop
        
        while len(permutations) < self.num_permutations and attempts < max_attempts:
            perm = self.identity.copy()
            random.shuffle(perm)
            perm_tuple = tuple(perm)
            if perm_tuple not in seen:
                permutations.append(perm)
                seen.add(perm_tuple)
            attempts += 1
        
        if len(permutations) < self.num_permutations:
            print(f"Only generated {len(permutations)} unique permutations out of requested {self.num_permutations}.")
        
        # Sort permutations by similarity to identity (descending)
        permutations_sorted = sorted(
            permutations,
            key=lambda p: compute_similarity(p, self.identity),
            reverse=True
        )
        
        return permutations_sorted
    
    def get_permutation(self, rotation_in_degrees: float) -> List[int]:
        """
        Get the permutation corresponding to the rotation_in_degrees.
        
        Parameters:
        - rotation_in_degrees (float): Rotation degrees between 0 and 180.
        
        Returns:
        - List[int]: Permuted labels list.
        """
        # Clamp rotation_in_degrees to [0, 180]
        rotation_in_degrees = rotation_in_degrees % 360
        rotation_in_degrees = min(rotation_in_degrees, 360 - rotation_in_degrees)
        
        # Calculate permutation index
        # Ensuring that rotation_in_degrees=0 maps to index=0 (identity)
        # and rotation_in_degrees=180 maps to index=num_permutations-1 (most shuffled)
        permutation_index = int((rotation_in_degrees / 180) * (self.num_permutations - 1))
        permutation_index = min(permutation_index, self.num_permutations - 1)
        
        return self.permutations[permutation_index]



def load_permuted_labels_mnist_data(
    batch_size: int = 32,
    rotation_in_degrees: float = 0,
    download: bool = True,
    root: str = 'data',
    num_workers: int = 0,
    num_permutations: int = 100,
    ):
    """
    Load the MNIST dataset with labels permuted based on rotation_in_degrees.
    
    Parameters:
    - batch_size (int): Number of samples per batch.
    - rotation_in_degrees (float): Rotation degrees between 0 and 180.
    - num_permutations (int): Number of permutations to generate.
    - download (bool): Whether to download the dataset.
    - root (str): Directory to store the dataset.
    - num_workers (int): Number of subprocesses for data loading.
    
    Returns:
    - Tuple[DataLoader, DataLoader]: DataLoaders for training and testing with permuted labels.
    """
    # Define the label list (e.g., 0-9 for MNIST)
    label_list = list(range(10))
    
    # Initialize the PermutedLabelMapper
    mapper = PermutedLabelMapper(label_list, num_permutations)
    
    # Get the permutation based on rotation_in_degrees
    perm = mapper.get_permutation(rotation_in_degrees)
    
    # Create a mapping dictionary: original_label -> permuted_label
    mapping = {original: permuted for original, permuted in zip(label_list, perm)}
    
    # Define the target_transform to apply the permutation
    def target_transform(label):
        return mapping[label]
    
    # Define image transformations: ToTensor, Normalize, and Flatten
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])
    
    # Load the training dataset with the transformations and target_transform
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform,
        target_transform=target_transform
    )
    
    # Load the test dataset with the transformations and target_transform
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform,
        target_transform=target_transform
    )
    
    # Create DataLoaders for training and testing datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def load_rotated_flattened_mnist_data(batch_size=32, rotation_in_degrees=0, download=True, root=DATA_DIR, num_workers=0):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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


def load_permuted_flattened_mnist_data(batch_size=32, permutation_percent=0, download=True, root=DATA_DIR, num_workers=1):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

def show_and_save_permuted_labels():
    import matplotlib.pyplot as plt
        # Plot the images with pixel permutations
    rotations = list(range(0, 360, 36))
    fig, axes = plt.subplots(len(rotations), 10, figsize=(15, 1.5 * len(rotations)))

    for i, rotation in enumerate(rotations):
        # Parameters
        batch_size = 64
        rotation_in_degrees = rotation  # Example: 90 degrees rotation
        num_permutations = 100
        download = True
        root = 'data'  # Change as needed
        num_workers = 2

        # Load the permuted MNIST data
        train_loader, _ = load_permuted_labels_mnist_data(
            batch_size=batch_size,
            rotation_in_degrees=rotation_in_degrees,
            num_permutations=num_permutations,
            download=download,
            root=root,
            num_workers=num_workers
        )

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

    plt.savefig(f'./rotated_labels_examples.png')



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
    show_and_save_permuted_labels()
