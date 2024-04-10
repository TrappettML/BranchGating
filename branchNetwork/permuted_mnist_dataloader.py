import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def blockwise_permutation(image, block_size=4):
    """
    Permute an image by blocks of size `block_size x block_size`.
    
    Parameters:
    - image: The input image as a 2D numpy array.
    - block_size: The size of the block to permute within.
    
    Returns:
    - The permuted image as a 2D numpy array.
    """
    # Ensure the image dimensions are divisible by block_size
    h, w = image.shape
    assert h % block_size == 0 and w % block_size == 0, "Image dimensions must be divisible by block_size."
    
    permuted_image = np.zeros_like(image)
    permutation_indices = np.random.permutation(h * w // (block_size ** 2))
    
    block_index = 0
    for permuted_index in permutation_indices:
        orig_row = (block_index // (w // block_size)) * block_size
        orig_col = (block_index % (w // block_size)) * block_size
        
        perm_row = (permuted_index // (w // block_size)) * block_size
        perm_col = (permuted_index % (w // block_size)) * block_size
        
        permuted_image[perm_row:perm_row+block_size, perm_col:perm_col+block_size] = image[orig_row:orig_row+block_size, orig_col:orig_col+block_size]
        block_index += 1
    
    return permuted_image


def show_and_save_permuted_mnist_examples():
    """
    Load MNIST data, find one example of each digit, show and save the figure,
    with each row showing the digits with a different permutation.
    """
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    
    # Find one example of each digit
    digit_images = [None] * 10
    found_digits = set()
    for images, labels in train_loader:
        for i, label in enumerate(labels):
            if label.item() in found_digits:
                continue
            # Remove channel dimension and convert to numpy for easier manipulation
            digit_images[label.item()] = images[i].squeeze().numpy()
            found_digits.add(label.item())
            if len(found_digits) == 10:
                break
        if len(found_digits) == 10:
            break

    # Plot and save the images with permutations
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))  # 4 rows for 4 permutations
    for row in range(1):
        permutation = np.random.permutation(784)  # Random permutation
        for col, img in enumerate(digit_images):
            ax = axes[row, col]
            # permuted_img = img.flatten()[permutation].reshape(28, 28)  # Apply permutation and reshape back
            ax.imshow(img, cmap='gray')
            if row == 0:
                ax.set_title(str(col))
            ax.axis('off')
    for row in range(1,4):
        permutation = np.random.permutation(784)  # Random permutation
        for col, img in enumerate(digit_images):
            ax = axes[row, col]
            permuted_img = img.flatten()[permutation].reshape(28, 28)  # Apply permutation and reshape back
            ax.imshow(permuted_img, cmap='gray')
            if row == 0:
                ax.set_title(str(col))
            ax.axis('off')

    plt.savefig('permuted_mnist_examples.png')


if __name__=='__main__':
    show_and_save_permuted_mnist_examples()