import torch
import torch.nn as nn
import torch.nn.functional as F
from src.res_block import ResBlock
from src.self_attention import SelfAttention



class SplatUNetDiffu(nn.Module):
    def __init__(self, in_channels=36, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        self.inc = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1_res = ResBlock(64, time_dim); self.down1_conv = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2_res = ResBlock(128, time_dim); self.down2_conv = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bot1 = ResBlock(256, time_dim); self.attn = SelfAttention(256); self.bot2 = ResBlock(256, time_dim)
        self.up1_scale = nn.Upsample(scale_factor=2, mode='nearest'); self.up1_conv = nn.Conv2d(256, 128, 3, padding=1); self.fuse1 = nn.Conv2d(256, 128, 1); self.up1_res = ResBlock(128, time_dim)
        self.up2_scale = nn.Upsample(scale_factor=2, mode='nearest'); self.up2_conv = nn.Conv2d(128, 64, 3, padding=1); self.fuse2 = nn.Conv2d(128, 64, 1); self.up2_res = ResBlock(64, time_dim)
        self.outc = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type_as(x) / 1000.0
        t_emb = self.time_mlp(t)
        x1 = self.inc(x)
        x_skip1 = self.down1_res(x1, t_emb); x2 = self.down1_conv(x_skip1)
        x_skip2 = self.down2_res(x2, t_emb); x3 = self.down2_conv(x_skip2)
        x_bot = self.attn(self.bot1(x3, t_emb)); x_bot = self.bot2(x_bot, t_emb)
        x_up = self.up1_res(self.fuse1(torch.cat([self.up1_conv(self.up1_scale(x_bot)), x_skip2], dim=1)), t_emb)
        x_up = self.up2_res(self.fuse2(torch.cat([self.up2_conv(self.up2_scale(x_up)), x_skip1], dim=1)), t_emb)
        return torch.tanh(self.outc(x_up))