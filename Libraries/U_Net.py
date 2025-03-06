import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from numpy import ndarray
from typing import Callable
import logging, time, math
from .Utils import *

logger = logging.getLogger(__name__)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, activation: nn.Module = nn.GELU()) -> None:
        super(Down, self).__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, in_channels, residual=True, activation=activation),
            DoubleConv(in_channels, out_channels, activation=activation)
        )
        self.time_seq = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.seq(x)
        t_emb = self.time_seq(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t_emb

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,time_embed_dim: int, activation: nn.Module = nn.GELU()) -> None:
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.half_channels = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.seq = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, activation=activation),
            DoubleConv(in_channels, out_channels, in_channels // 2, activation=activation)
        )
        self.time_seq = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
    def forward(self, x: Tensor, x_skip: Tensor, t: Tensor) -> Tensor:
        x = torch.cat([x, x_skip], dim=1)
        x = self.half_channels(x)
        x = self.upsample(x)
        x = self.seq(x)
        t_emb = self.time_seq(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t_emb

class Attention(nn.Module):
    def __init__(self, channels: int, size: int, activation: nn.Module = nn.GELU()) -> None:
        super(Attention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.seq = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            activation,
            nn.Linear(channels, channels)
        )
    def forward(self, x: Tensor) -> None:
        _, _, H, W = x.shape
        patch: int = self.size
        n_patches: int = (H * W) // patch

        x = x.view(-1, self.channels, n_patches * patch).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_val, _ = self.mha(x_ln, x_ln, x_ln)
        attention_val = attention_val + x
        attention_val = attention_val + self.seq(attention_val)
        return attention_val.swapaxes(2, 1).view(-1, self.channels, H, W)

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False, activation: nn.Module = nn.GELU()) -> None:
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.activation = activation
        if mid_channels is None:
            mid_channels = out_channels
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            self.activation, 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            return self.activation(x + self.seq(x))
        else: 
            return self.seq(x)

class SE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, activation: nn.Module = nn.GELU()) -> None:
        super(SE, self).__init__()
        self.channels = in_channels
        red_channels = max(in_channels // reduction, 1)
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, red_channels, kernel_size=1),
            activation,
            nn.Conv2d(red_channels, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid()  
        )
    def forward(self, x: Tensor) -> Tensor:
        se = self.seq(x)
        return x * se

class Attention_U_NET(nn.Module):
    def __init__(self, in_channels: int = 1, time_embed_dim: int = 256, n_starting_filters: int = 32, n_starting_attention_size = 32, n_downsamples: int = 3, activation: nn.Module = nn.GELU(), device: str = "cpu") -> None:
        super(Attention_U_NET, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        self.inp_lay = nn.Conv2d(in_channels, n_starting_filters, kernel_size=3, stride=1, padding=1)
        self.encoder: list = []
        for i in range(n_downsamples):
            down_seq = nn.Sequential(
                Down(n_starting_filters * (2 ** i), n_starting_filters * (2 ** (i + 1)), time_embed_dim, activation),
                Attention(n_starting_filters * (2 ** (i + 1)), n_starting_attention_size // (2 ** i), activation)
                )
            self.encoder.append(down_seq)

        self.bottleneck = nn.Sequential(
            DoubleConv(n_starting_filters * (2 ** n_downsamples), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** n_downsamples), activation=activation)
        )

        self.decoder: list = []
        for i in reversed(range(n_downsamples)):
            up_seq = nn.Sequential(
                Up(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** i), time_embed_dim, activation),
                Attention(n_starting_filters * (2 ** i), (n_starting_attention_size // 2) * (2 ** (n_downsamples - i)), activation)

            )
            self.decoder.append(up_seq)

        self.out_lay = nn.Conv2d(n_starting_filters, in_channels, kernel_size=1)

    
    def time_embed(self, t: Tensor, channels: int) -> Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device) / channels))
        enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([enc_a, enc_b], dim=1)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        skip_sols: list = []
        t = t.unsqueeze(-1)
        t = self.time_embed(t, self.time_embed_dim)

        x = self.inp_lay(x)
        skip_sols.append(x)

        for block in self.encoder:
            x = block[0](x, t)
            x = block[1](x)
            skip_sols.append(x)

        x = self.bottleneck(x)
        skip_sols = skip_sols[::-1]

        for i, block in enumerate(self.decoder):
            x = block[0](x, skip_sols[i], t)
            x = block[1](x)

        return self.out_lay(x)
    
    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "Encoder:\n"
        for i, module in enumerate(self.encoder):
            s += f"  Encoder[{i}]: {module}\n"
        s += "Bottleneck:\n"
        s += f"  {self.bottleneck}\n"
        s += "Decoder:\n"
        for i, module in enumerate(self.decoder):
            s += f"  Decoder[{i}]: {module}\n"
        return s

class SE_U_NET(nn.Module):
    def __init__(self, in_channels: int = 1, time_embed_dim: int = 256, n_starting_filters: int = 32, n_starting_se_red = 16, n_downsamples: int = 3, activation: nn.Module = nn.GELU(), device: str = "cpu") -> None:
        super(SE_U_NET, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        self.inp_lay = nn.Conv2d(in_channels, n_starting_filters, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList()
        for i in range(n_downsamples):
            down_seq = nn.Sequential(
                Down(n_starting_filters * (2 ** i), n_starting_filters * (2 ** (i + 1)), time_embed_dim, activation),
                SE(n_starting_filters * (2 ** (i + 1)), n_starting_se_red * (2 ** i), activation)
                )
            self.encoder.append(nn.ModuleList(down_seq))

        self.bottleneck = nn.Sequential(
            DoubleConv(n_starting_filters * (2 ** n_downsamples), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** n_downsamples), activation=activation)
        )

        self.decoder = nn.ModuleList()
        for i in reversed(range(n_downsamples)):
            up_seq = nn.Sequential(
                Up(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** i), time_embed_dim, activation),
                SE(n_starting_filters * (2 ** i), n_starting_se_red * (2 ** (n_downsamples - i)), activation)
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
            x = block[1](x)
            skip_sols.append(x)
        x = self.bottleneck(x)
        skip_sols = skip_sols[::-1]

        for i, block in enumerate(self.decoder):
            x = block[0](x, skip_sols[i], t)
            x = block[1](x)
        out: Tensor = self.out_lay(x)
        return out
    
    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "Encoder:\n"
        for i, module in enumerate(self.encoder):
            s += f"  Encoder[{i}]: {module}\n"
        s += "Bottleneck:\n"
        s += f"  {self.bottleneck}\n"
        s += "Decoder:\n"
        for i, module in enumerate(self.decoder):
            s += f"  Decoder[{i}]: {module}\n"
        return s

class Conv_U_NET(nn.Module):
    def __init__(self, in_channels: int = 1, time_embed_dim: int = 256, n_starting_filters: int = 32, n_downsamples: int = 3, activation: nn.Module = nn.GELU(), device: str = "cpu") -> None:
        super(Conv_U_NET, self).__init__()
        self.device = device
        self.time_embed_dim = time_embed_dim
        self.inp_lay = nn.Conv2d(in_channels, n_starting_filters, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList()
        for i in range(n_downsamples):
            down_seq = nn.Sequential(
                Down(n_starting_filters * (2 ** i), n_starting_filters * (2 ** (i + 1)), time_embed_dim, activation),
                DoubleConv(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** (i + 1)), residual=True, activation=activation)
                )
            self.encoder.append(nn.ModuleList(down_seq))

        self.bottleneck = nn.Sequential(
            DoubleConv(n_starting_filters * (2 ** n_downsamples), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** (n_downsamples + 1)), activation=activation),
            DoubleConv(n_starting_filters * (2 ** (n_downsamples + 1)), n_starting_filters * (2 ** n_downsamples), activation=activation)
        )

        self.decoder = nn.ModuleList()
        for i in reversed(range(n_downsamples)):
            up_seq = nn.Sequential(
                Up(n_starting_filters * (2 ** (i + 1)), n_starting_filters * (2 ** i), time_embed_dim, activation),
                DoubleConv(n_starting_filters * (2 ** i), n_starting_filters * (2 ** i), residual=True, activation=activation)
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
            x = block[1](x)
            skip_sols.append(x)
        x = self.bottleneck(x)
        skip_sols = skip_sols[::-1]

        for i, block in enumerate(self.decoder):
            x = block[0](x, skip_sols[i], t)
            x = block[1](x)
        out: Tensor = self.out_lay(x)
        return out
    
    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "Encoder:\n"
        for i, module in enumerate(self.encoder):
            s += f"  Encoder[{i}]: {module}\n"
        s += "Bottleneck:\n"
        s += f"  {self.bottleneck}\n"
        s += "Decoder:\n"
        for i, module in enumerate(self.decoder):
            s += f"  Decoder[{i}]: {module}\n"
        return s
