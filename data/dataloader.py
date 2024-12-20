import torch
from torchvision import datasets, transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # Bypass SSL verification

def get_dataloaders(batch_size, num_workers, cuda):
    train_transforms = transforms.Compose([
        # Random rotation (e.g., rotate images by -10 to +10 degrees)
        transforms.RandomRotation(degrees=15),
    
        # Random horizontal flip with 50% probability
        #transforms.RandomHorizontalFlip(p=0.5),
    
        # Random affine transformation with slight shifts, rotation, and scaling
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    
        # Convert to tensor and normalize as before
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size)

    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

    return train_loader, test_loader

