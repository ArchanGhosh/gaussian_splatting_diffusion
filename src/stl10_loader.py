import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from src.config import DEVICE


def get_stl10_dataloader(batch_size=32, image_size=128, target_class = 0, split="train", data_root="./data"):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    print(f"\n{'-'*10} Downloading STL 10 data {'-'*10}")
    dataset = datasets.STL10(root=data_root, split=split, download=True, transform=transform)
    dataset.data = dataset.data[:5000]

    subset_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]

    dataloader = DataLoader(Subset(dataset, subset_indices),
                            batch_size = batch_size,
                            shuffle=True,
                            drop_last=True)
    
    return dataloader

#loader = get_stl10_dataloader(batch_size=4, image_size=128)