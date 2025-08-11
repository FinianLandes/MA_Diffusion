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
    def __init__(self, in_channels: int, intermediate_channels: int, kernel_size: int = 3, dilation: int = 1, res_scale: float = 1.0) -> None:
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, intermediate_channels, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(intermediate_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )
    def forward(self, x: Tensor) -> Tensor:
        return x + self.res_scale * self.conv(x)

class ResNet(nn.Module):
    def __init__(self, in_channels: int, n_depth: int, m_conv: float = 1.0, dilation_growth_rate: int = 1, dilation_cycle: (int | None) = None,  res_scale: bool = False, reverse_dilation: bool = False):
        super(ResNet, self).__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResBlock(in_channels, int(m_conv * in_channels), dilation=dilation_growth_rate ** _get_depth(depth), res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res_blocks(x)
        return x
    
class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down_t: int, stride_t: int, n_depth: int = 8, m_conv: float = 1.0, dilation_growth_rate: int = 3) -> None:
        super(EncoderConvBlock, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(current_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU()
            )
        )
        current_channels = 32
        for i in range(down_t):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, out_channels, kernel_size=3, stride=stride_t, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            current_channels = out_channels
        self.resnet = ResNet(out_channels, n_depth=n_depth, m_conv=m_conv, dilation_growth_rate=dilation_growth_rate)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.resnet(x)
        return x

class DecoderConvBlock(nn.Module):
    def __init__(self, input_emb_width: int, output_emb_width: int, down_t: int, stride_t: int, n_depth: int = 8, m_conv: float = 1.0, dilation_growth_rate: int = 3, dilation_cycle: Optional[int] = None, reverse_dilation: bool = False) -> None:
        super().__init__()
        blocks = []
        current_channels = input_emb_width
        for i in range(down_t):
            blocks.append(
                nn.Sequential(
                    ResNet(
                        current_channels,
                        n_depth=n_depth,
                        m_conv=m_conv,
                        dilation_growth_rate=dilation_growth_rate,
                        dilation_cycle=dilation_cycle,
                        reverse_dilation=reverse_dilation
                    ),
                    nn.ConvTranspose1d(
                        current_channels,
                        output_emb_width if i < down_t - 1 else input_emb_width,
                        kernel_size=3,
                        stride=stride_t,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm1d(output_emb_width if i < down_t - 1 else input_emb_width),
                    nn.ReLU()
                )
            )
            current_channels = output_emb_width if i < down_t - 1 else input_emb_width
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_emb_width: int, output_emb_width: int, levels: int, downs_t: list, strides_t: list, n_depth: int = 8, m_conv: float = 1.0, dilation_growth_rate: int = 3) -> None:
        super(Encoder, self).__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.level_blocks = nn.ModuleList()
        for level, down_t, stride_t in zip(range(levels), downs_t, strides_t):
            in_channels = input_emb_width if level == 0 else output_emb_width
            self.level_blocks.append(
                EncoderConvBlock(
                    in_channels=in_channels,
                    out_channels=output_emb_width,
                    down_t=down_t,
                    stride_t=stride_t,
                    n_depth=n_depth,
                    m_conv=m_conv,
                    dilation_growth_rate=dilation_growth_rate
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        xs = []
        current_x = x
        for _, (level_block, _, _) in enumerate(zip(self.level_blocks, self.downs_t, self.strides_t)):
            current_x = level_block(current_x)
            xs.append(current_x)
        return xs[-1] if self.levels == 1 else xs

class Decoder(nn.Module):
    def __init__(self, input_emb_width: int, output_emb_width: int, levels: int, downs_t: list, strides_t: list, n_depth: int = 8, m_conv: float = 1.0, dilation_growth_rate: int = 3, dilation_cycle: Optional[int] = None, reverse_dilation: bool = False) -> None:
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.level_blocks = nn.ModuleList()
        for level, down_t, stride_t in zip(range(levels), downs_t, strides_t):
            self.level_blocks.append(
                DecoderConvBlock(
                    input_emb_width=output_emb_width,
                    output_emb_width=output_emb_width,
                    down_t=down_t,
                    stride_t=stride_t,
                    n_depth=n_depth,
                    m_conv=m_conv,
                    dilation_growth_rate=dilation_growth_rate,
                    dilation_cycle=dilation_cycle,
                    reverse_dilation=reverse_dilation
                )
            )
        self.out = nn.Conv1d(output_emb_width, input_emb_width, kernel_size=3, stride=1, padding=1)

    def forward(self, xs: list, all_levels: bool = True) -> Tensor:
        x = xs[-1]
        for level, level_block in enumerate(reversed(self.level_blocks)):
            x = level_block(x)
            if all_levels and level < self.levels - 1:
                x = x + xs[self.levels - level - 2]
        x = self.out(x)
        return x
    
class BottleneckBlock(nn.Module):
    def __init__(self, k_bins: int, emb_width: int, mu: float = 0.99, threshold: float = 1.0) -> None:
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.threshold = threshold
        self.reset_k()

    def reset_k(self) -> None:
        self.init = False
        self.k_sum = None
        self.k_elem = None
        self.register_buffer('k', torch.zeros(self.k_bins, self.emb_width))

    def _tile(self, x: Tensor) -> Tensor:
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.02 / np.sqrt(ew) 
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_k(self, x: Tensor) -> None:
        self.init = True
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][:self.k_bins]
        self.k = _k_rand
        self.k_sum = self.k.clone()
        self.k_elem = torch.ones(self.k_bins, device=self.k.device)

    def update_k(self, x: Tensor, x_l: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            x_l_onehot = torch.zeros(self.k_bins, x.shape[0], device=x.device)
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)
            _k_sum = torch.matmul(x_l_onehot, x)
            _k_elem = x_l_onehot.sum(dim=-1)
            y = self._tile(x)
            _k_rand = y[torch.randperm(y.shape[0])][:self.k_bins]
            old_k = self.k.clone()
            self.k_sum = self.mu * self.k_sum + (1. - self.mu) * _k_sum
            self.k_elem = self.mu * self.k_elem + (1. - self.mu) * _k_elem
            usage = (self.k_elem >= self.threshold).float()
            self.k = usage.view(-1, 1) * (self.k_sum / self.k_elem.view(-1, 1)) + (1 - usage).view(-1, 1) * _k_rand
            _k_prob = _k_elem / torch.sum(_k_elem)
            entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))
            used_curr = (_k_elem >= self.threshold).sum()
            dk = torch.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, dk=dk)

    def preprocess(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.permute(0, 2, 1).contiguous().view(-1, self.emb_width)
        prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        return x, prenorm

    def postprocess(self, x_l: (Tensor | None), x_d: Tensor, N: int, T: int) -> tuple[(Tensor | None), Tensor]:
        x_l = x_l.view(N, T) if x_l is not None else None
        x_d = x_d.view(N, T, self.emb_width).permute(0, 2, 1).contiguous()
        return x_l, x_d

    def quantise(self, x: Tensor) -> tuple[Tensor, Tensor]:
        k_w = self.k.t()
        distance = torch.sum(x**2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w**2, dim=0, keepdim=True)
        _, x_l = torch.min(distance, dim=-1)
        fit = torch.mean(torch.min(distance, dim=-1).values)
        return x_l, fit

    def dequantise(self, x_l: Tensor) -> Tensor:
        x_d = F.embedding(x_l, self.k)
        return x_d

    def forward(self, x: Tensor, update_k: bool = True) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        N, width, T = x.shape
        assert width == self.emb_width, f"Expected input channels {self.emb_width}, got {width}"
        x, prenorm = self.preprocess(x)
        if update_k and not self.init:
            self.init_k(x)
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)
        if update_k and self.training:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}
        commit_loss = torch.norm(x_d.detach() - x)**2 / np.prod(x.shape)
        x_d = x + (x_d - x).detach()
        x_l, x_d = self.postprocess(x_l, x_d, N, T)
        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)

class VQVAE(nn.Module):
    def __init__(self, input_emb_width: int = 1, output_emb_width: int = 64, k_bins: int = 2048, levels: int = 1, downs_t: list = [3], strides_t: list = [2]):
        super().__init__()
        self.encoder = Encoder(input_emb_width, output_emb_width, levels, downs_t, strides_t)
        self.bottleneck = BottleneckBlock(k_bins, output_emb_width, mu=0.99)
        self.decoder = Decoder(input_emb_width, output_emb_width, levels, downs_t, strides_t)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        z = self.encoder(x)
        z_indicies, z_quantized, commit_loss, metrics = self.bottleneck(z)
        x_recon = self.decoder([z_quantized], all_levels=False)
        return x_recon, z_indicies, commit_loss, metrics
    
    def decode(self, x: (Tensor | list)) -> Tensor:
        x_l = x if isinstance(x, Tensor) else x[0]
        z_dq = self.bottleneck.dequantise(x_l) 
        N, T = x_l.shape[0], x_l.shape[1]
        _, z_dq = self.bottleneck.postprocess(None, z_dq, N, T)
        audio = self.decoder([z_dq], all_levels=False)
        return audio
    
    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z_indicies, _, _, _ = self.bottleneck(z)
        return z_indicies.detach().cpu().long()
