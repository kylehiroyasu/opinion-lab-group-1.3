import torch
from torch import nn

class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()
    
    def forward(self, x):
        return x