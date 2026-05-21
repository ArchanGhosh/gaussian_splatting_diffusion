import torch.nn as nn
import torch.nn.functional as F


from src.config import TOTAL_CHANNELS

class SplatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 128x128 -> Output 32x32 (Factor 4 downsampling)
        self.net = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),

            # Refinement at 32x32
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),

            # Project to Splat Parameters (36 Channels)
            nn.Conv2d(256, TOTAL_CHANNELS, 1)
        )

    def forward(self, x):
        # We assume input x is already resized to 128x128 or we resize it here
        if x.shape[-1] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return self.net(x)