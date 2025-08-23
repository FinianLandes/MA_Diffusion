# Torch
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F

# Utils
import numpy as np
from numpy import ndarray
import logging, math

# Base Scripts
from .Utils import *

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.lay = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU()
    def forward(self, x: Tensor) -> Tensor:
        out = self.lay(x) + x
        return self.act(out)
    
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1,downsample_facotrs: list = [2, 2, 2, 2], base_channels: int = 64, latent_dim: int = 32, n_down: int = 4) -> None:
        super().__init__()
        layers = []
        channels = base_channels
        layers.append(nn.Conv1d(in_channels, channels, 7, padding=3))
        for i in range(n_down):
            layers.append(ResBlock(channels))
            layers.append(nn.Conv1d(channels, channels * 2, 4, stride=downsample_facotrs[i], padding=1))
            channels *= 2
        self.layers = nn.Sequential(*layers)
        self.mu = nn.Conv1d(channels, latent_dim, kernel_size=1)
        self.logvar = nn.Conv1d(channels, latent_dim, kernel_size=1)
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.layers(x)
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1, downsample_factors: list = [2, 2, 2, 2], base_channels: int = 64, latent_dim: int = 32, n_up: int = 4) -> None:
        super().__init__()
        layers = []
        channels = base_channels *(2 ** n_up)
        downsample_factors.reverse()
        layers.append(nn.Conv1d(latent_dim, channels, 3, padding=1))
        for i in range(n_up):
            layers.append(ResBlock(channels))
            layers.append(nn.Upsample(scale_factor=downsample_factors[i], mode="linear", align_corners=False))
            layers.append(nn.Conv1d(channels, channels // 2, 3, padding=1))
            channels //= 2
        layers.append(nn.Conv1d(channels, out_channels, 7, padding=3))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        x_reconst = self.layers(z)
        return x_reconst

class VAE(nn.Module):
    def __init__(self, in_channels: int = 1, downsample_factors: list = [2, 2, 2, 2], base_channels: int = 64, latent_dim: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        assert len(downsample_factors) == n_layers, "Downsample Factors needs to have the same length as n_layers"
        self.encoder = Encoder(in_channels, downsample_factors, base_channels, latent_dim, n_layers)
        self.decoder = Decoder(in_channels, downsample_factors, base_channels, latent_dim, n_layers)
        self.out_act = nn.Tanh()

    def encode(self, x: Tensor) -> tuple[Tensor,Tensor]:
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        x_reconst = self.decoder(z)
        return self.out_act(x_reconst)
    
    def forward(self, x: Tensor) -> tuple[Tensor,...]:
        mu, logvar = self.encode(x)

        std = torch.exp(0.5 * logvar)
        eps =  torch.randn_like(std)
        z_rep = mu + std * eps

        x_recost = self.decode(z_rep)
        return x_recost, mu, logvar
    


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(256, 512, 128), win_lengths=(1024, 2048, 512)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            X = torch.stft(x.squeeze(1), n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=torch.hann_window(win_length, device=x.device), return_complex=True)
            Y = torch.stft(y.squeeze(1), n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=torch.hann_window(win_length, device=y.device), return_complex=True)

            mag_X = torch.abs(X)
            mag_Y = torch.abs(Y)

            sc_loss = torch.norm(mag_X - mag_Y, p=1) / torch.norm(mag_Y, p=1)
            mag_loss = F.l1_loss(mag_X, mag_Y)

            loss += sc_loss + mag_loss

        return loss / len(self.fft_sizes)