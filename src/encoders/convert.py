import torch
from torch import nn


class NormalizeRGB(nn.Module):

    def __init__(self):
        super(NormalizeRGB, self).__init__()
        self.means = nn.Parameter(torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.stdevs = nn.Parameter(torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, data):
        return (data - self.means) / self.stdevs
