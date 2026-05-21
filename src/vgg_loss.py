import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]  
        self.slice3 = vgg[9:16] 


        for param in self.parameters():
            param.requires_grad = False

        self.eval()
        self.to(DEVICE)


        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE))

    def forward(self, x, y):

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std


        h_x1 = self.slice1(x); h_y1 = self.slice1(y)
        h_x2 = self.slice2(h_x1); h_y2 = self.slice2(h_y1)
        h_x3 = self.slice3(h_x2); h_y3 = self.slice3(h_y2)


        loss = F.mse_loss(h_x1, h_y1) + F.mse_loss(h_x2, h_y2) + F.mse_loss(h_x3, h_y3)
        return loss