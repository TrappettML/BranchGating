
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import socket


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


def show_and_save_rotated_mnist_examples(rotation_angles):
    import matplotlib.pyplot as plt
    """
    Load MNIST data, find one example of each digit, show and save the figure,
    with each row showing the digits rotated by the corresponding angle in rotation_angles.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    
    # Find one example of each digit
    digit_images = [None] * 10
    found_digits = set()
    for images, labels in train_loader:
        for i, label in enumerate(labels):
            if label.item() in found_digits:
                continue
            digit_images[label.item()] = images[i].squeeze()  # Remove channel dimension
            found_digits.add(label.item())
            if len(found_digits) == 10:
                break
        if len(found_digits) == 10:
            break

    # Plot the images with rotations
    num_angles = len(rotation_angles)
    fig, axes = plt.subplots(num_angles, 10, figsize=(15, 1.5 * num_angles))
    for i, angle in enumerate(rotation_angles):
        for j, img in enumerate(digit_images):
            ax = axes[i, j] if num_angles > 1 else axes[j]
            rotated_img = rotate_image(img, angle)  # Rotate the image
            ax.imshow(rotated_img, cmap='gray')
            if i == 0:
                ax.set_title(str(j))
            ax.axis('off')

    plt.savefig(f'/home/users/MTrappett/mtrl/BranchGatingProject/data/rotated_mnist_examples.png')
    plt.show()

def rotate_image(image, angle):
    """
    Rotate an image (2D numpy array) by a given angle.
    """
    from scipy.ndimage import rotate
    return rotate(image.numpy(), angle, reshape=False)


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
        'KMNIST': datasets.KMNIST
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


def load_permuted_flattened_mnist_data(batch_size=32, permute_seed=42, download=True, root=DATA_DIR):
    import numpy as np
    """
    Load the MNIST dataset with each image's pixels permuted according to a given seed and flattened.

    Parameters:
    - batch_size: The number of samples per batch to load.
    - permute_seed: Seed used to generate a consistent permutation of the pixels.
    - download: Whether to download the dataset if not locally available.
    - root: Directory where the dataset will be stored.

    Returns:
    - train_loader: DataLoader for the permuted and flattened training data.
    - test_loader: DataLoader for the permuted and flattened test data.
    """

    # Define transformation pipeline
    if permute_seed != 0:
        # Set the random seed for reproducibility of the permutation
        np.random.seed(permute_seed)
        idx = torch.from_numpy(np.random.permutation(28 * 28))  # Create a permutation of the 784 pixel indices
        
        # Define the transformation with permutation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)[idx].view(1, 28, 28)),  # Permute the pixels
            transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
            transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
        ])
    else:
        # Define the transformation without permutation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
            transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
        ])

    # Load the training and test datasets with the specified transforms
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    # Create data loaders for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_permuted_mnist_by_seeds(seeds, root=DATA_DIR, save_path='permuted_mnist.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    # Define a basic transform to convert image data to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    # Prepare a figure to hold the subplots
    fig, axes = plt.subplots(nrows=len(seeds), ncols=10, figsize=(15, 1.5 * len(seeds)))
    if len(seeds) == 1:
        axes = np.expand_dims(axes, 0)  # Handle the case of a single row

    # Collect one example of each digit
    examples = {i: None for i in range(10)}
    for image, label in dataset:
        if examples[label] is None:
            examples[label] = image
        if all(ex is not None for ex in examples.values()):
            break

    # Plot each permuted set of images
    for row_idx, seed in enumerate(seeds):
        if seed is not None:
            np.random.seed(seed)
            idx = np.random.permutation(28 * 28)
        else:
            idx = np.arange(28 * 28)

        for digit, image in examples.items():
            if seed is not None:
                # Permute the image pixels based on the seed
                permuted_image = image.view(-1)[torch.tensor(idx)].view(1, 28, 28)
            else:
                permuted_image = image

            ax = axes[row_idx, digit]
            ax.imshow(permuted_image.numpy().squeeze(), cmap='gray')
            ax.set_title(f"Seed {seed} - Digit {digit}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # show_and_save_rotated_mnist_examples([0, 120, 180, 240])
    # print('Data saved')
    seeds = [None, 42, 123, 456]
    plot_permuted_mnist_by_seeds(seeds)
    print('Data saved')
    
def test_load_rotated_flattened_mnist():
    train_loader, test_loader = load_rotated_flattened_mnist_data(batch_size=128, rotation_in_degrees=120)
    print(f'Length of train_loader: {len(train_loader)}')
    print(f'Length of test_loader: {len(test_loader)}')
    for data, target in train_loader:
        print(data.shape)
        print(target.shape)
        break

    for data, target in test_loader:
        print(data.shape)
        print(target.shape)
        break

    print('Data loaded successfully')

def test_load_permuted_flattened_mnist():
    train_loader, test_loader = load_permuted_flattened_mnist_data(batch_size=128, permute_seed=42)
    print(f'Length of train_loader: {len(train_loader)}')
    print(f'Length of test_loader: {len(test_loader)}')
    for data, target in train_loader:
        print(data.shape)
        print(target.shape)
        break

    for data, target in test_loader:
        print(data.shape)
        print(target.shape)
        break

    print('Data loaded successfully')
    
    
if __name__=='__main__':
    # test_load_rotated_flattened_mnist()
    test_load_permuted_flattened_mnist()
    main()
