import torch
from src.config import _ALPHA_HAT

def noise_images(x, t):
    """Add noise to Splat Grids"""
    sqrt_alpha_hat = torch.sqrt(_ALPHA_HAT[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - _ALPHA_HAT[t])[:, None, None, None]
    epsilon = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon