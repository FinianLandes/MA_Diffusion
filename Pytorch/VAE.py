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

class ConvBottleneckEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.conv_mean = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(in_channels, latent_dim, kernel_size=1) 
    def forward(self, x: Tensor) -> tuple[Tensor,...]:
        return self.conv_mean(x), self.conv_logvar(x)

class ConvBottleneckDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.expand = nn.ConvTranspose2d(latent_dim, out_channels, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:
        return self.expand(z)

class LinBottleneckEncoder(nn.Module):
    def __init__(self, flattened_size: int, latent_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hid_mean = nn.Linear(flattened_size, latent_dim)
        self.hid_logvar = nn.Linear(flattened_size, latent_dim)
    def forward(self, x: Tensor) -> tuple[Tensor,...]:
        x = self.flatten(x)
        return self.hid_mean(x), self.hid_logvar(x)

class LinBottleneckDecoder(nn.Module):
    def __init__(self, latent_dim: int, flattened_size: int):
        super().__init__()
        self.fc_decode = nn.Linear(latent_dim, flattened_size)
    def forward(self, x: Tensor) -> Tensor:
        return self.fc_decode(x)

class ConvDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation):
        super(ConvDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x

class ConvUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation):
        super(ConvUp, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            activation,
            nn.Upsample(scale_factor=2, mode="nearest")
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, device: str = "cpu", activation = nn.LeakyReLU(0.3), input_shape: ndarray = [0, 1, 2048, 128], n_conv_blocks: int = 2, n_starting_filters: int = 32, lin_bottleneck: bool = False) -> None:
        """Class for Convolutional VAE
        Args:
            in_channels (int): Number of input channels.
            latent_dim (int): Size of latent Dimension (1-D).
            device (str): Computing Device. Defaults to cpu.
            activation (_type_, optional): A torch activation fucntion. Defaults to nn.LeakyReLU(0.3).
            input_shape (ndarray, optional): shape of input in format [B, C, H, W]. Defaults to [0, 1, 2048, 128].
            n_conv_blocks (int, optional): Number of convoltional blocks (Conv -> Activation -> Pool) size reduction is  // (4 ** number of blocks) + 2. Defaults to 2.
            n_starting_filters (int, optional): Filter double in each Conv Block. Defaults to 32.
            lin_bottleneck (bool, optional): Whether The bottleneck is made form convolutional layers or linear layers, if convolutional latent dim is a number of channels otherwise size of the lin layer. Defaults to False.
        Returns:
            None
        """
        super(VAE, self).__init__()
        self.device = device
        self.activation = activation
        self.input_shape = input_shape
        self.n_starting_filters = n_starting_filters 
        self.n_divs = n_conv_blocks * 2 + 1
        self.n_conv_blocks = n_conv_blocks
        #encoder
        layers: list = []
        for i in range(self.n_conv_blocks):
            if i == 0:
                layers.append(ConvDown(in_channels, self.n_starting_filters, activation))
            else:
                layers.append(ConvDown(self.n_starting_filters * (2 ** (i - 1)), self.n_starting_filters * (2 ** i), activation))
        layers.append(nn.Conv2d(self.n_starting_filters * (2 ** (self.n_conv_blocks - 1)), self.n_starting_filters * (2 ** self.n_conv_blocks), kernel_size=4, stride=2, padding=1))
        layers.append(activation)
        self.encoder = nn.Sequential(*layers)

        self.encoded_w, self.encoded_h = input_shape[-1] // (2 ** self.n_divs), input_shape[-2] // (2 ** self.n_divs)
        flattened_size = self.n_starting_filters * (2 ** self.n_conv_blocks) * self.encoded_h * self.encoded_w
        if lin_bottleneck:
            self.bottleneck_encoder = LinBottleneckEncoder(flattened_size, latent_dim)
            self.bottleneck_decoder = LinBottleneckDecoder(latent_dim, flattened_size)
        else:
            self.bottleneck_encoder = ConvBottleneckEncoder(self.n_starting_filters * (2 ** self.n_conv_blocks), latent_dim)
            self.bottleneck_decoder = ConvBottleneckDecoder(latent_dim, self.n_starting_filters * (2 ** self.n_conv_blocks))
        layers = []
        for i in range(self.n_conv_blocks, 0, -1):
            layers.append(ConvUp(self.n_starting_filters * (2 ** i), self.n_starting_filters * (2 ** (i - 1)), activation))
        layers.append(nn.ConvTranspose2d(self.n_starting_filters, in_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        mean, logvar = self.bottleneck_encoder(x)
        return mean, logvar
    
    def reparam(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mean + eps * std
    
    def decode(self, z: Tensor) -> Tensor:
        x: Tensor = self.bottleneck_decoder(z)
        x = x.view(-1, self.n_starting_filters * (2 ** self.n_conv_blocks), self.encoded_h, self.encoded_w) 
        x = self.decoder(x)
        return x
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, var = self.encode(x)
        z = self.reparam(mean, var)
        x_pred = self.decode(z)
        return x_pred, mean, var

def loss_VAE(x: Tensor, x_pred: Tensor, mean: Tensor, logvar: Tensor, alpha: float) -> tuple[Tensor, Tensor, Tensor]:
    reprod_loss = torch.sqrt(nn.functional.mse_loss(x_pred, x, reduction="sum"))
    eps = 1e-8
    KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - (logvar + eps).exp())
    return alpha * reprod_loss + KL, reprod_loss * alpha, KL

def train_VAE(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, loss_function: callable, epochs: int, device: str, reprod_loss_weight: float, checkpoint_freq: int = 0, model_path: str = "") -> list[float]:
    """Function for training a VAE from VAE.py

    Args:
        model (nn.Module): Torch VAE model.
        data_loader (DataLoader): Torch Dataloader.
        optimizer (optim.Optimizer): Torch Optimizer.
        loss_function (callable): Loss function for training.
        epochs (int): Number of training Epochs.
        device (str): A torch device eg cpu.
        reprod_loss_weight (float): Weight of the reprod loss vs the KL divergence.
        checkpoint_freq (int, optional): Saves the model every n epochs set to 0 for no saving. Defaults to 0.
        model_path (str, optional): Model path with the suffix .pth for where to save the model. Defaults to "".

    Returns:
        list[float]: Loss of each epoch.
    """
    model.train()
    logger.info(f"Training started on {device}")
    total_time: float = 0
    loss_list: list = []
    for e in range(epochs):
        total_loss: float = 0
        total_reprod: float = 0
        total_KL: float = 0
        start_time: float = time.time()
        for batch_idx, (x, _) in enumerate(data_loader):
            with torch.autocast(device):
                x: Tensor = x.to(device)
                x = x.unsqueeze(1)
                x_pred, mean, logvar = model(x)
                loss, reprod_loss, KL = loss_function(x, x_pred, mean, logvar, reprod_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_reprod += reprod_loss.item()
            total_KL += KL.item()
            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Batch {batch_idx + 1:02d}/{len(data_loader):02d}", end='')
        if logger.getEffectiveLevel() == LIGHT_DEBUG:
            print()

        avg_loss = total_loss / len(data_loader.dataset)
        avg_reprod = total_reprod / len(data_loader.dataset)
        avg_KL = total_KL / len(data_loader.dataset)
        epoch_time = time.time() - start_time
        total_time += epoch_time
        remaining_time = int((total_time / (e + 1)) * (epochs - e - 1))

        logger.info(f"Epoch {e + 1:02d}: Avg. Loss: {avg_loss:.5e} Avg. Reprod: {avg_reprod:.5e} Avg. KL: {avg_KL:.5e} Remaining Time: {remaining_time // 3600:02d}h {(remaining_time % 3600) // 60:02d}min {round(remaining_time % 60):02d}s")
        loss_list.append(avg_loss)
        if checkpoint_freq > 0 and (e + 1) % checkpoint_freq == 0:
            torch.save(model.state_dict(), model_path)
            logger.light_debug(f"Checkpoint saved model to {model_path}")
    return loss_list

def generate_sample(model: VAE, device: str, sample: Tensor = None, num_samples: int = 1, lin_bottleneck: bool = False) -> ndarray:
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
