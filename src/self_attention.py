import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).swapaxes(1, 2)
        x_ln = self.ln(x_flat)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        attn = attn + x_flat
        out = self.ff_self(attn) + attn
        return out.swapaxes(2, 1).view(B, C, H, W)
