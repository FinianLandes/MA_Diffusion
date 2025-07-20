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

class X_Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dim_ff: int, max_seq_len: int, init_scale: float = 0.02) -> None:
        super(X_Transformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.layers = nn.ModuleList([SparseTransformerLayer(d_model, n_heads, dim_ff)
            for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, mean=0.0, std=init_scale)
        if torch.isnan(self.embed.weight).any():
            print("NaN detected in embed.weight, reinitializing...")
        nn.init.normal_(self.pos_embed, mean=0.0, std=init_scale * 2)
        nn.init.normal_(self.out.weight, mean=0.0, std=init_scale)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = self.embed(x)
        x = x * math.sqrt(self.d_model)
        x = x + self.pos_embed[:seq_len].unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int) -> None:
        super().__init__()
        self.self_attn = AxisAlignedAttention(d_model, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class AxisAlignedAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super(AxisAlignedAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        block_size = 64
        if seq_len % block_size != 0:
            raise ValueError(f"seq_len {seq_len} must be divisible by block_size {block_size}")
        n_blocks = seq_len // block_size

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.n_head, 3 * self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        x_2d = x.view(batch_size, n_blocks, block_size, -1).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        qkv_2d = self.qkv(x_2d)
        qkv_2d = qkv_2d.view(batch_size, seq_len, self.n_head, 3 * self.head_dim).transpose(1, 2)
        q_2d, k_2d, v_2d = qkv_2d.chunk(3, dim=-1)

        row_mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(x.device)
        row_mask = row_mask.repeat(1, 1, n_blocks, n_blocks)
        row_attn = self._attention(q, k, v, mask=row_mask)

        col_mask = torch.triu(torch.ones(n_blocks, n_blocks), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(x.device)
        col_mask = col_mask.repeat(1, 1, block_size, block_size)
        col_attn = self._attention(q_2d, k_2d, v_2d, mask=col_mask)

        prev_row_mask = torch.ones(n_blocks, n_blocks).bool().unsqueeze(0).unsqueeze(0).to(x.device)
        prev_row_mask = prev_row_mask & (torch.arange(n_blocks).unsqueeze(1) < torch.arange(n_blocks).unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(x.device)
        prev_row_mask = prev_row_mask.repeat(1, 1, block_size, block_size)
        prev_row_attn = self._attention(q, k, v, mask=prev_row_mask)

        attn_output = (row_attn + col_attn + prev_row_attn) / 3
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out(attn_output)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        return attn @ v


class TransformerDec(nn.Module):
    def __init__(self, vocab_size: int = 2048, embed_size: int = 512, n_layers: int = 6, forward_expansion: int = 4, n_heads: int = 8, pad_idx: int = -1, dropout: float = 0.1, device: str = "cpu", max_seq_len: int = 2048) -> None:
        super(TransformerDec, self).__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.embed_size = embed_size
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_seq_len, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, n_heads, dropout, forward_expansion, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight, gain=1.0)

    def make_mask(self, x: Tensor) -> Tensor:
        N, seq_len = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).expand(N, 1, seq_len, seq_len).to(self.device)
        return mask.bool()

    def forward(self, x: Tensor) -> Tensor:
        N, seq_len = x.shape
        mask = self.make_mask(x)
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embed(x) * math.sqrt(self.embed_size) + self.pos_embed(pos))
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, n_heads: int) -> None:
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        assert self.head_dim * n_heads == embed_size, "Embed size needs to be multiple of n_heads"

        self.v = nn.Linear(embed_size, embed_size, bias=True)
        self.k = nn.Linear(embed_size, embed_size, bias=True)
        self.q = nn.Linear(embed_size, embed_size, bias=True)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        nn.init.xavier_uniform_(self.v.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k.weight, gain=1.0)
        nn.init.xavier_uniform_(self.q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)

    def forward(self, v: Tensor, k: Tensor, q: Tensor, mask: Tensor | None = None) -> Tensor:
        N = q.shape[0]
        v_len, k_len, q_len = v.shape[1], k.shape[1], q.shape[1]

        v = self.v(v).reshape(N, v_len, self.n_heads, self.head_dim)
        k = self.k(k).reshape(N, k_len, self.n_heads, self.head_dim)
        q = self.q(q).reshape(N, q_len, self.n_heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        if mask is not None:
            energy = energy.masked_fill(~mask.bool(), float("-inf"))
        attn = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attn, v]).reshape(N, q_len, self.n_heads * self.head_dim)
        out = self.norm(self.fc_out(out))
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, n_heads: int, dropout: float, forward_expansion: int, device: str = "cpu") -> None:
        super(DecoderBlock, self).__init__()
        self.attn = SelfAttention(embed_size, n_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attn = self.attn(x, x, x, mask)
        x = self.dropout(self.norm1(attn + x))
        fwd = self.ff(x)
        x = self.dropout(self.norm2(fwd + x))
        return x