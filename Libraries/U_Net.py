import torch
from torch import nn, Tensor, functional
import logging, math


logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    def __init__(self, channels: int, t_embed: int,  kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

        self.act = nn.LeakyReLU(0.2)
        self.time_proj = nn.Linear(t_embed, channels)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_embed = self.time_proj(t).unsqueeze(-1)
        out = self.conv1(x)
        out = self.act(out + t_embed)
        out = self.conv2(out)
        return self.act(out + x)

class UNet(nn.Module):
    def __init__(self, in_channels: int, n_layers: int, base_channels: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        layers = []
        channels = base_channels

        layers.append(nn.Conv1d(in_channels, channels, 7, padding=3))
        for _ in range(n_layers):
            layers.append(ResBlock(channels, embed_dim))
            layers.append(nn.Conv1d(channels, channels * 2, 4, stride=2, padding=1))
            channels *= 2
        self.down_layers = nn.ModuleList(layers)

        self.bottleneck = ResBlock(channels, embed_dim)

        layers = []

        for _ in range(n_layers):
            layers.append(nn.ConvTranspose1d(channels, channels // 2, 4, 2, 1))
            layers.append(nn.Conv1d(channels, channels // 2, 3, padding=1))
            layers.append(ResBlock(channels // 2, embed_dim))
            channels //= 2
        layers.append(nn.Conv1d(channels, in_channels, 7, padding=3))
        self.up_layers = nn.ModuleList(layers)

        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                    nn.SiLU(),
                                    nn.Linear(embed_dim, embed_dim)
                                    )

    def time_embed(self, t: Tensor, embed_dim: int) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]

        half_dim = embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=t.device).float() / half_dim)

        angles = t.float() * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb

    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_mlp(self.time_embed(t, self.embed_dim))

        skips = []
        for layer in self.down_layers:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
                skips.append(x)
            else:
                x = layer(x)

        x = self.bottleneck(x, t_emb)

        skips.reverse()
        i = 0
        for layer in self.up_layers:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, nn.ConvTranspose1d):
                x = layer(x)
                x = torch.cat([x, skips[i]], dim=1)
                i += 1
            else:
                x = layer(x)
        return x