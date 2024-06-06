import torch
from torch import nn


class ImageLogScaling(nn.Module):
    def __init__(self, vmin, vmax):
        super().__init__()
        self.vmin = vmin#self.register_buffer('vmin', torch.tensor([vmin], dtype=torch.float32))
        self.vmax = vmax#self.register_buffer('vmax', torch.tensor([vmax], dtype=torch.float32))

    def forward(self, image):
        image = (torch.log(image) - self.vmin) / (self.vmax - self.vmin)
        return image
