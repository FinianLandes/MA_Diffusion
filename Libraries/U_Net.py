import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import numpy as np
from numpy import ndarray
from typing import Callable
import logging, time
from .Utils import *

logger = logging.getLogger(__name__)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, activation: nn.Module = nn.GELU(),use_modulation: bool = True) -> None:
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.seq = nn.ModuleList([
            DoubleConv(in_channels, in_channels, residual=True, activation=activation, use_modulation=use_modulation),
            DoubleConv(in_channels, out_channels, activation=activation, use_modulation=use_modulation)
        ])
        self.time_seq = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.pool(x)
        for  layer in self.seq:
            x = layer(x, t)
        t_emb = self.time_seq(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t_emb

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, activation: nn.Module = nn.GELU(), use_modulation: bool = True) -> None:
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.half_channels = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.seq = nn.ModuleList([
            DoubleConv(in_channels, in_channels, residual=True, activation=activation, use_modulation=use_modulation),
            DoubleConv(in_channels, out_channels, in_channels // 2, activation=activation, use_modulation=use_modulation)
        ])
        self.time_seq = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
    def forward(self, x: Tensor, x_skip: Tensor, t: Tensor) -> Tensor:
        x = torch.cat([x, x_skip], dim=1)
        x = self.half_channels(x)
        x = self.upsample(x)
        for  layer in self.seq:
            x = layer(x, t)
        t_emb = self.time_seq(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t_emb

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, time_embed_dim: int = 128, residual: bool = False, activation: nn.Module = nn.GELU(), use_modulation: bool = True) -> None:
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.activation = activation
        self.use_modulation = use_modulation
        if mid_channels is None:
            mid_channels = out_channels
        self.module_list = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            Modulation(mid_channels, time_embed_dim) if use_modulation else None,
            self.activation, 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            Modulation(out_channels, time_embed_dim) if use_modulation else None,
            self.activation
        ]
        self.seq = nn.ModuleList([layer for layer in self.module_list if layer is not None])
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        identity: Tensor = x
        for layer in self.seq:
            if isinstance(layer, Modulation):
                x = layer(x, t)
            else:
                x = layer(x)
        if self.residual:
            return x + identity
        else: 
            return x

class Modulation(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int) -> None:
        super(Modulation, self).__init__()
        self.scale = nn.Linear(time_embed_dim, channels)
        self.shift = nn.Linear(time_embed_dim, channels)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        scale = self.scale(t).view(-1, x.shape[1], 1, 1)
        shift = self.shift(t).view(-1, x.shape[1], 1, 1)
        return x * (1 + scale) + shift

class Conv_U_NET(nn.Module):
    def __init__(self, in_channels: int = 1, time_embed_dim: int = 256, n_starting_filters: int = 32, n_downsamples: int = 3, activation: nn.Module = nn.GELU(), use_modulation: bool = False,  device: str = "cpu") -> None:
        super(Conv_U_NET, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        self.inp_lay = nn.Conv2d(in_channels, n_starting_filters, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList()
        for i in range(n_downsamples):
            down_seq = nn.Sequential(
                Down(n_starting_filters * (2 ** i), n_starting_filters * (2 ** (i + 1)), time_embed_dim, activation, use_modulation=use_modulation),
                DoubleConv(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** (i + 1)), residual=True, activation=activation, use_modulation=use_modulation)
                )
            self.encoder.append(nn.ModuleList(down_seq))

        self.bottleneck = nn.ModuleList([
            DoubleConv(n_starting_filters * (2 ** n_downsamples), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation, use_modulation=use_modulation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation, use_modulation=use_modulation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** n_downsamples), activation=activation, use_modulation=use_modulation)
        ])

        self.decoder = nn.ModuleList()
        for i in reversed(range(n_downsamples)):
            up_seq = nn.Sequential(
                Up(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** i), time_embed_dim, activation, use_modulation=use_modulation),
                DoubleConv(n_starting_filters * (2 ** i), n_starting_filters * (2 ** i), residual=True, activation=activation, use_modulation=use_modulation)
            )
            self.decoder.append(nn.ModuleList(up_seq))

        self.out_lay = nn.Conv2d(n_starting_filters, in_channels, kernel_size=1)

    
    def time_embed(self, t: Tensor, channels: int) -> Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([enc_a, enc_b], dim=1)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        skip_sols: list = []
        if t.dim() == 1:
            t = t.unsqueeze(-1).type(torch.float)
        t = self.time_embed(t, self.time_embed_dim)

        x = self.inp_lay(x)

        for block in self.encoder:
            x = block[0](x, t)
            x = block[1](x, t)
            skip_sols.append(x)
        
        for layer in self.bottleneck:
            x = layer(x, t)

        skip_sols = skip_sols[::-1]

        for i, block in enumerate(self.decoder):
            x = block[0](x, skip_sols[i], t)
            x = block[1](x, t)
        out: Tensor = self.out_lay(x)
        return out
    
    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "Encoder:\n"
        s += f"Input Layer: {self.inp_lay}\n"
        for i, module in enumerate(self.encoder):
            s += f"  Encoder[{i}]: {module}\n"
        s += "Bottleneck:\n"
        s += f"  {self.bottleneck}\n"
        s += "Decoder:\n"
        for i, module in enumerate(self.decoder):
            s += f"  Decoder[{i}]: {module}\n"
        s += f"Output Layer: {self.out_lay}\n"
        return s

class Threshold_LR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, thresholds: list, lr: list, last_epoch: int = -1, verbose: bool = False) -> None:
        self.thresholds = thresholds
        self.lr = lr
        self.current_loss = float('inf')
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> list:
        for i, threshold in enumerate(self.thresholds):
            if self.current_loss > threshold:
                lr_idx = max(0, i - 1)
                break
            else:
                lr_idx = len(self.lr) - 1 

        return [self.lr[lr_idx] for _ in self.optimizer.param_groups]
    
    def step(self, loss: float = None, epoch: int = None) -> None:
        if loss is not None:
            self.current_loss = loss
        super().step(epoch)