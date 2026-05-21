import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.config import DEVICE


def get_stl10_dataloader(batch_size=32, image_size=128, split="train", data_root="./data"):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.STL10(root=data_root, split=split, download=True, transform=transform)
    dataset.data = dataset.data[:5000]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#loader = get_stl10_dataloader(batch_size=4, image_size=128)