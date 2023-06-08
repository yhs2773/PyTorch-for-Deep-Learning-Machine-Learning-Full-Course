"""
Contains functionality for creating PyTorch DataLoader
for image classification data
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """
    Creates training and testing dataloaders

    Takes in training and testing directory paths and turns them
    into PyTorch Dataset and then into PyTorch DataLoader

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms to perform on training and testing data
        batch_size: Number of samples per batch in each of the DataLoader
        num_workers: An integer for number of workers per DataLoader
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        where class_names is a list of the target classes
        
        Example usage:
            train_dataloader, test_dataloader, class_names = \
                create_dataloaders(train_dir=path/to/train_dir,
                                   test_dir=path/to/test_dir,
                                   transform=some_transform,
                                   batch_size=32,
                                   num_workers=2)
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
