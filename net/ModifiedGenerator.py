import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import LeakyReLU
from net.block import *
from net.block import _equalized_conv2d
from net.SGFMT import TransformerModel
from net.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from net.CMSFFT import ChannelTransformer

class modifiedGenerator(nn.Module):
    def __init__(self,original_generator):
        super(modifiedGenerator, self).__init__()
        self.model = original_generator

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.model(x)
        return x