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

class ConditionalNorm(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int = 128, num_groups: int = 8) -> None:
        super(ConditionalNorm, self).__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(num_groups, channels)
        self.t_weight = nn.Linear(time_embed_dim, channels)
        self.t_bias = nn.Linear(time_embed_dim, channels)
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.norm(x)
        weight: Tensor = self.t_weight(t)
        bias: Tensor = self.t_bias(t)
        x = x * weight.view(-1, self.channels, 1, 1) + bias.view(-1, self.channels, 1, 1)
        return x



class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 128, activation: nn.Module = nn.GELU(), conditional_norm: bool = False) -> None:
        super(Down, self).__init__()
        self.cond_norm = conditional_norm
        self.seq = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, in_channels, residual=True, time_embed_dim=time_embed_dim, activation=activation, conditional_norm=conditional_norm),
            DoubleConv(in_channels, out_channels, time_embed_dim=time_embed_dim, activation=activation, conditional_norm=conditional_norm)
        ])
        self.time_seq = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        for layer in self.seq:
            if isinstance(layer, DoubleConv):
                x = layer(x, t)
            else:
                x = layer(x)
        if not self.cond_norm:
            t_emb = self.time_seq(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x = x + t_emb
        return x

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 128, activation: nn.Module = nn.GELU(), conditional_norm: bool = False) -> None:
        super(Up, self).__init__()
        self.cond_norm = conditional_norm
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.half_channels = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.seq = nn.Sequential(
            self.half_channels,
            self.upsample,
            DoubleConv(in_channels, in_channels, residual=True, time_embed_dim=time_embed_dim, activation=activation, conditional_norm=conditional_norm),
            DoubleConv(in_channels, out_channels, in_channels // 2, time_embed_dim=time_embed_dim, activation=activation, conditional_norm=conditional_norm)
        )
        self.time_seq = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        for layer in self.seq:
            if isinstance(layer, DoubleConv):
                x = layer(x, t)
            else:
                x = layer(x)
        if not self.cond_norm:
            t_emb = self.time_seq(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x = x + t_emb
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False, time_embed_dim: int = 128, activation: nn.Module = nn.GELU(), conditional_norm: bool = False) -> None:
        super(DoubleConv, self).__init__()
        self.cond_norm = conditional_norm
        self.residual = residual
        self.activation = activation
        if mid_channels is None:
            mid_channels = out_channels
        self.seq = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            ConditionalNorm(mid_channels, time_embed_dim=time_embed_dim, num_groups=8) if self.cond_norm else nn.GroupNorm(8, mid_channels),
            self.activation, 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ConditionalNorm(out_channels, time_embed_dim=time_embed_dim, num_groups=8) if self.cond_norm else nn.GroupNorm(8, out_channels),
            self.activation
        ])
    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        identity: Tensor = x
        for layer in self.seq:
            if isinstance(layer, ConditionalNorm):
                x = layer(x, t)
            else:
                x = layer(x)

        if self.residual:
            return x + identity
        else: 
            return x

class Conv_U_NET(nn.Module):
    def __init__(self, in_channels: int = 1, time_embed_dim: int = 256, n_starting_filters: int = 32, n_downsamples: int = 3, activation: nn.Module = nn.GELU(), conditional_norm: bool = False,  device: str = "cpu") -> None:
        super(Conv_U_NET, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        self.inp_lay = nn.Conv2d(in_channels, n_starting_filters, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList()
        for i in range(n_downsamples):
            down_seq = nn.Sequential(
                Down(n_starting_filters * (2 ** i), n_starting_filters * (2 ** (i + 1)), activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm),
                DoubleConv(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** (i + 1)), residual=True, activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm)
                )
            self.encoder.append(nn.ModuleList(down_seq))

        self.bottleneck = nn.ModuleList([
            DoubleConv(n_starting_filters * (2 ** n_downsamples), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** (n_downsamples + 1)), residual=True, activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** n_downsamples), activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm)
        ])

        self.decoder = nn.ModuleList()
        for i in reversed(range(n_downsamples)):
            up_seq = nn.Sequential(
                Up(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** i), activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm),
                DoubleConv(n_starting_filters * (2 ** i), n_starting_filters * (2 ** i), residual=True, activation=activation, time_embed_dim=time_embed_dim, conditional_norm=conditional_norm)
            )
            self.decoder.append(nn.ModuleList(up_seq))

        self.out_lay = nn.Conv2d(n_starting_filters * 2, in_channels, kernel_size=1)

    
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
        skip_sols.append(x)

        for block in self.encoder:
            x = block[0](x, t)
            x = block[1](x, t)
            skip_sols.append(x)
        
        for layer in self.bottleneck:
            x = layer(x, t)

        skip_sols = skip_sols[::-1]

        for i, block in enumerate(self.decoder):
            x = torch.cat([x, skip_sols[i]], 1)
            x = block[0](x, t)
            x = block[1](x, t)
        
        x = torch.cat([x, skip_sols[-1]], dim=1)
        out: Tensor = self.out_lay(x)
        return out
    
    def __repr__(self) -> str:
        s = "Encoder:\n"
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

class EMA():
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self. decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
        
    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]