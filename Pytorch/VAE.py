import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from numpy import ndarray
import logging
import time
from Utils import *

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, device: str, activation = nn.LeakyReLU(0.3), input_shape: ndarray = [0, 0, 2048, 128]) -> None:
        super(VAE, self).__init__()
        self.device = device
        self.activation = activation
        self.input_shape = input_shape
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), #-> B, 32, H/2, W/2
            activation,
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1), #-> B, 32, H/4, W/4
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), #-> B, 64, H/8, W/8
            activation,
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1), #-> B, 64, H/16, W/16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #-> B, 128, H/32, W/32
            activation
        )
        self.encoded_w, self.encoded_h = input_shape[-1] // 32, input_shape[-2] // 32
        flattened_size = 128 * self.encoded_h * self.encoded_w

        self.flatten = nn.Flatten() # -> B, 128 * H/32 * W/32
        self.hid_mean = nn.Linear(flattened_size, latent_dim) # -> B, Latent Dim
        self.hid_var = nn.Linear(flattened_size, latent_dim) # -> B, Latent Dim
        #decoder
        self.fc_decode = nn.Linear(latent_dim, flattened_size) # -> B, 128 * H/32 * W/32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> B, 64, H/16, W/16
            activation,
            nn.Upsample(scale_factor=2, mode="nearest"),# -> B, 64, H/8, W/8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> B, 32, H/4, W/4
            activation,
            nn.Upsample(scale_factor=2, mode="nearest"), # -> B, 32, H/2, W/2
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # -> B, 1, H, W
            nn.Sigmoid()
        )


    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        x = self.flatten(x)
        mean, logvar = self.hid_mean(x), self.hid_var(x)
        return mean, logvar
    
    def reparam(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mean + eps * std
    
    def decode(self, z: Tensor) -> Tensor:
        x: Tensor = self.fc_decode(z)
        x = x.view(-1, 128, self.encoded_h, self.encoded_w) 
        x = self.decoder(x)
        return x
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, var = self.encode(x)
        z = self.reparam(mean, var)
        x_pred = self.decode(z)
        return x_pred, mean, var

def loss_VAE(x: Tensor, x_pred: Tensor, mean: Tensor, logvar: Tensor, alpha: float) -> Tensor:
    reprod_loss = torch.sqrt(nn.functional.mse_loss(x_pred, x, reduction="sum"))
    eps = 1e-8
    KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - (logvar + eps).exp())
    return alpha * reprod_loss + KL

def train_VAE(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, loss_function: callable, epochs: int, device: str, reprod_loss_weight: float) -> float:    
    model.train()
    logger.info(f"Training started on {device}")
    
    for e in range(epochs):
        total_loss: float = 0
        start_time: float = time.time()
        for batch_idx, (x, _) in enumerate(data_loader):
            x: Tensor = x.to(device)
            x = x.unsqueeze(1)
            x_pred, mean, logvar = model(x)
            loss: torch.Tensor = loss_function(x, x_pred, mean, logvar, reprod_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Batch {batch_idx + 1:02d}/{len(data_loader):02d}", end='')
        if logger.getEffectiveLevel() == LIGHT_DEBUG:
            print()

        avg_loss = total_loss / len(data_loader.dataset)
        epoch_time = time.time() - start_time
        remaining_time = int(epoch_time * (epochs - e - 1))

        logger.info(f"Epoch {e + 1:02d}: Avg. Loss: {avg_loss:.5e} Remaining Time: {remaining_time // 3600:02d}h {(remaining_time % 3600) // 60:02d}min {round(remaining_time % 60):02d}s")
    return total_loss

def generate_sample(model: VAE, device: str, sample: Tensor = None, num_samples: int = 1) -> ndarray:
    model.eval()
    logger.light_debug("Started creating samples")
    with torch.no_grad():
        if sample is not None:
            mean, var = model.encode(sample.to(device))
            std = torch.exp(0.5 * var)
            eps = torch.randn((num_samples,) + mean.shape, device=device)
            z = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        else:
            z = torch.randn((num_samples, model.hid_mean.out_features), device=device)
        x_pred = model.decode(z).cpu().numpy() 
    logger.light_debug(f"Created samples: {x_pred.shape}")
    return x_pred

def fwd_pass(model: VAE, device: str, sample: Tensor = None) -> ndarray:
    model.eval()
    logger.light_debug("Started passthrough")
    with torch.no_grad():
        mean, logvar = model.encode(sample.to(device))
        z = model.reparam(mean, logvar)
        x = model.decode(z)
    x = x.cpu().numpy() 
    logger.light_debug(f"Created samples: {x.shape}")
    return x
