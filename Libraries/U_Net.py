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

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 1, activation: nn.Module = nn.GELU(), width_down_sampling: int = 2) -> None:
        super(ResBlock, self).__init__()
        width_down_sampling = width_down_sampling if factor > 1 else 1
        self.activation = activation
        self.in_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=factor, padding=1) if factor > 1 else nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        identity = x
        x = self.layers(x)
        x = self.activation(x + identity)
        return self.downsample(x)
class Simple_Embed(nn.Module):
    def __init__(self,  channels: int, embed_dim: int = 128) -> None:
        super(Simple_Embed, self).__init__()
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                embed_dim,
                channels
            ),
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        embed = self.embed_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x += embed
        return x

class Modulation(nn.Module):
    def __init__(self, channels: int, embed_dim: int = 128) -> None:
        super(Modulation, self).__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels, eps=1e-06)
        self.weight = nn.Linear(embed_dim, channels)
        self.bias = nn.Linear(embed_dim, channels)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x = self.norm(x)
        w: Tensor = self.weight(emb) + 1
        b: Tensor = self.bias(emb)
        x = x * w.view(-1, self.channels, 1, 1) + b.view(-1, self.channels, 1, 1)
        return x

class Attention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 12, features: int = 64) -> None:
        super(Attention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_features = features
        self.total_features = num_heads * features

        self.norm = nn.InstanceNorm2d(channels)
        self.to_qvk = nn.Conv2d(channels, self.total_features * 3, kernel_size=1, bias=False)
        self.scale = self.head_features ** -0.5
        self.out_proj = nn.Conv2d(self.total_features, channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        b,c,h,w = x.shape
        qkv: Tensor = self.to_qvk(x)
        qkv = qkv.view(b, 3, self.num_heads, self.head_features, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum('bnhi,bnhj->bnhij', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bnhij,bnhj->bnhi', attn, v) 
        out = out.view(b, self.num_heads * self.head_features, h, w)
        out = self.out_proj(out)
        x = x + out
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, channels: int, expansion_factor: int = 4, activation: nn.Module = nn.GELU()) -> None:
        super(FeedForward, self).__init__()
        self.channels = channels

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * expansion_factor, kernel_size=1),
            activation, 
            nn.Conv2d(channels * expansion_factor, channels, kernel_size=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.ffn(x)
        return x + identity

class Injection(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, activation: nn.Module = nn.GELU()) -> None:
        super(Injection, self).__init__()
        self.layers = nn. Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            activation
        )
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = torch.cat([x, skip], dim=1)
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int, n_res_blocks: int, attention: bool, attn_heads: int, attn_features: int, activation: nn.Module, embed_dim: int, up: bool = False, width_upsample: int = 2, embed_class: nn.Module = Modulation) -> None:
        super(Block, self).__init__()
        if n_res_blocks == 0:
            n_res_blocks = 1
        width_upsample = width_upsample if factor > 1 else 1
        self.init_res = ResBlock(in_channels, in_channels if up else out_channels, 1, activation)
        self.upsample = nn.Sequential(ResBlock(in_channels, in_channels, 1, activation), 
                                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=factor, padding=1)) if up else nn.Identity()
        self.injection = Injection(in_channels, in_channels, out_channels, activation) if up else nn.Identity()
        self.attention_layers = nn.ModuleList([
                                            embed_class(out_channels, embed_dim), 
                                            Attention(out_channels, attn_heads, attn_features) if attention else nn.Identity(), 
                                            FeedForward(out_channels, 4) if attention else nn.Identity()])
        self.res_blocks = nn.ModuleList([ResBlock(out_channels, out_channels, 1, activation) for _ in range(n_res_blocks - 1)])
        self.downsample = ResBlock(out_channels, out_channels, factor, activation) if not up else nn.Identity()
            

    def forward(self, x: Tensor, t: Tensor, skip: Tensor = None) -> Tensor:
        logger.debug(f"Block fwd start Size: {x.shape}")
        x = self.upsample(x)
        logger.debug(f"After upsample Size: {x.shape}")
        x = self.init_res(x)
        logger.debug(f"After Init res Size: {x.shape}")
        if skip is not None:
            skip = self.upsample[1](skip)
            logger.debug(f"After skip Size: {x.shape}")
            x = self.injection(x, skip)
            logger.debug(f"After inj. Size: {x.shape}")
        
        for layer in self.res_blocks:
            x = layer(x)
            logger.debug(x.shape, type(layer))

        for layer in self.attention_layers:
            if isinstance(layer, (Modulation, Simple_Embed)):
                x = layer(x, t)
            else:
                x = layer(x)
            logger.debug(x.shape, type(layer))
        
        return self.downsample(x)




class U_NET(nn.Module):
    def __init__(self, in_channels: int = 1, channels: list[int] = [256, 256, 512, 512, 1024, 1024], res_blocks: list[int] = [1, 2, 2, 2, 2, 2], factors: list[int] = [2, 2, 2, 2, 4, 4], attentions: list[int] = [0, 0, 0, 0, 0, 1], attention_heads: int = 12, attention_features: int = 64, activation: nn.Module = nn.GELU(), embeding_dim: int = 128, embed_class: nn.Module = Modulation, device: str = "cpu") -> None:
        super(U_NET, self).__init__()
        self.device = device
        self.time_embed_dim = embeding_dim
        channels.insert(0, in_channels)
        self.encoder = nn.ModuleList([Block(channels[i], channels[i + 1], factors[i], res_blocks[i], attentions[i], attention_heads, attention_features, activation, embeding_dim, embed_class=embed_class) for i in range(len(res_blocks))])
        channels.reverse()
        res_blocks.reverse()
        factors.reverse()
        attentions.reverse()
        self.bottleneck = nn.Sequential(ResBlock(channels[0], channels[0], 1, activation), ResBlock(channels[0], channels[0], 1, activation))
        self.decoder = nn.ModuleList([Block(channels[i], channels[i + 1], factors[i], res_blocks[i], attentions[i], attention_heads, attention_features, activation, embeding_dim, True,embed_class=embed_class) for i in range(len(res_blocks))])
    
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
        for layer in self.encoder:
            x = layer(x, t)
            skip_sols.append(x)
        x = self.bottleneck(x)
        skip_sols.reverse()
        for i, layer in enumerate(self.decoder):
            x = layer(x, t, skip_sols[i])
        return x
    
    def __repr__(self) -> str:
        s = "Encoder:\n"
        for i, module in enumerate(self.encoder):
            s += f"  Encoder[{i}]: {module}\n"
        s += "Bottleneck:\n"
        s += f"  {self.bottleneck}\n"
        s += "Decoder:\n"
        for i, module in enumerate(self.decoder):
            s += f"  Decoder[{i}]: {module}\n"
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

##################
### Deprecated ###
##################
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

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False, time_embed_dim: int = 128, activation: nn.Module = nn.GELU(), conditional_norm: bool = False) -> None:
        super(DoubleConv, self).__init__()
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

