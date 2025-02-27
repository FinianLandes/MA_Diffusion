import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from numpy import ndarray
import logging, time
from typing import Callable
from .Utils import *
from .U_Net import *

logger = logging.getLogger(__name__)

def linear_noise(T: int, beta_start: float = 0.0001,  beta_end: float = 0.02) -> Tensor:
    beta_t = torch.linspace(beta_start, beta_end, T)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)
    return alpha_bar_t

def cosine_noise(T: int, zero_offset: float = 0.00001) -> Tensor:
    steps = torch.linspace(0, T, T + 1)
    f_t = torch.cos((steps / T + zero_offset) / (1 + zero_offset) * (np.pi / 2)) ** 2
    alpha_bar_t = f_t / f_t[0]
    return alpha_bar_t


class Diffusion():
    def __init__(self, u_net: nn.Module, u_net_optimizer: optim.Optimizer, diffusion_timesteps: int) -> None:
        self.u_net: nn.Module = u_net
        self.optimizer: optim.Optimizer = u_net_optimizer
        self.T = diffusion_timesteps
    
    def embed_timestep(self, t: int) -> Tensor:
        ...
    
    def add_noise_t(t: int, x_0: Tensor, schedule: Tensor) -> Tensor:
        e: Tensor = torch.randn_like(x_0)
        return 

    def train(self, data_loader: DataLoader, device: str = "cpu", epochs: int = 100, loss_function: Callable = nn.SELU(), noise_schedule: Tensor = None, checkpoint_freq: int = 0, model_path: str = "") -> list[float]:
        loss_list: list[float] = []
        total_time: float = 0
        for e in range(epochs):
            total_loss: float = 0
            start_time: float = time.time()
            for b_idx, (x,_) in enumerate(data_loader):
                x: Tensor = x.to(device)
                e: Tensor = torch.randn_like(x)
                timesteps: Tensor = torch.randint(0, self.T, (x.shape[0],), device=device)
                t_embed: Tensor = self.embed_timestep(timesteps)
                x = torch.sqrt(noise_schedule[timesteps]) * x + torch.sqrt(1 - noise_schedule[timesteps]) * e

                pred_noise = self.u_net(x, t_embed)

                loss: Tensor = loss_function(pred_noise, e)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if logger.getEffectiveLevel() == LIGHT_DEBUG:
                    print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Batch {b_idx + 1:02d}/{len(data_loader):02d}", end='')
            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print()

            avg_loss = total_loss / len(data_loader.dataset)
            loss_list.append(avg_loss)

            epoch_time = time.time() - start_time
            total_time += epoch_time
            remaining_time = int((total_time / (e + 1)) * (epochs - e - 1))
            logger.info(f"Epoch {e + 1:02d}: Avg. Loss: {avg_loss:.5e} Remaining Time: {remaining_time // 3600:02d}h {(remaining_time % 3600) // 60:02d}min {round(remaining_time % 60):02d}s")

            if checkpoint_freq > 0 and (e + 1) % checkpoint_freq == 0:
                torch.save(self.u_net.state_dict(), model_path)
                logger.light_debug(f"Checkpoint saved model to {model_path}")
        return loss_list
