import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from numpy import ndarray
import logging, time
from Utils import *

logger = logging.getLogger(__name__)

class ConvBlockDown(nn.Module):
    def __init__(self, input_channels: int, activation = nn.LeakyReLU(0.3), n_groups: int = 8) -> None:
        super(ConvBlockDown, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=n_groups, num_channels=input_channels * 2),
            activation,
            nn.Conv2d(input_channels * 2, input_channels * 4, kernel_size=3, stride=1, padding=1),
            activation
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class ConvBlockUp(nn.Module):
    def __init__(self, input_channels: int, activation = nn.LeakyReLU(0.3)) -> None:
        super(ConvBlockUp, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1),
            activation,
            nn.ConvTranspose2d(input_channels // 2, input_channels // 4, kernel_size=3, stride=1, padding=1),
            activation
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class U_NET(nn.Module):
    def __init__(self, in_channels: int, device: str = "cpu", activation = nn.LeakyReLU(0.3), input_shape: ndarray = [0, 1, 2048, 128], n_res_layers: int = 2, n_starting_filters: int = 32, n_groups: int = 8) -> None:
        super(U_NET, self).__init__()
        self.device = device
        self.activation = activation
        self.n_groups = n_groups
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.n_res_layers = n_res_layers
        self.n_starting_filters = n_starting_filters

        layers: list = [nn.Sequential(
                nn.Conv2d(self.in_channels, n_starting_filters, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=self.n_groups, num_channels=n_starting_filters),
                activation,
                nn.Conv2d(n_starting_filters, n_starting_filters * 2, kernel_size=3, stride=1, padding=1),
                activation
            )
        ]
        for n in range(self.n_res_layers - 1):
            layers.append(ConvBlockDown(n_starting_filters * (2 ** (n + 1)), activation=activation, n_groups=self.n_groups))
        self.encoder = nn.ModuleList(layers)

        self.n_bottleneck_filters = n_starting_filters * (2 ** (self.n_res_layers + 1))
        layers: list = []
        for n in range(self.n_res_layers):
            layers.append(ConvBlockUp(self.n_bottleneck_filters // (2 ** (n)), activation=activation))
        self.decoder = nn.ModuleList(layers)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.n_bottleneck_filters // (2 ** (self.n_res_layers + 1)), 1, kernel_size=1, stride=1), 
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        skip_sols: list[Tensor] = []
        for block in self.encoder:
            x = block(x)
            skip_sols.append(x)
        for i, block in enumerate(self.decoder):
            x = block(x + skip_sols[-(i + 1)])
        x = self.final_conv(x)
        x = self.activation(x)
        return x

def train() -> list[float]:
    ...

def forward() -> list[float]:
    ...