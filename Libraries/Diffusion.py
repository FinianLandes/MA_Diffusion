import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
import numpy as np
from numpy import ndarray
import logging, time
from typing import Callable
from .Utils import *
from .U_Net import *

logger = logging.getLogger(__name__)


class Diffusion():
    def __init__(self, model: nn.Module, noise_steps: int, noise_schedule: str = "cosine", input_dim: list = [1, 1, 1024, 672], device: str = "cpu") -> None:
        self.model = model
        self.T = noise_steps
        self.inp_dim = input_dim
        self.device = device
        self.beta = self.get_noise_schedule(noise_schedule).to(self.device)[:, None, None, None]
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    
    def get_noise_schedule(self, noise_type: str = "cosine") -> Tensor:
        if noise_type == "cosine":
            return self.cosine_noise_schedule()
        if noise_type == "linear":
            return self.linear_noise_schedule()
        else:
            logger.fatal(f"Invalid Noise Schedule {noise_type}")
    
    def linear_noise_schedule(self, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
        return torch.linspace(beta_start, beta_end, self.T)
    
    def cosine_noise_schedule(self, zero_offset: float = 0.008) -> Tensor:
        steps = torch.linspace(0, self.T - 1, self.T, device=self.device)
        f_t = torch.cos((steps / self.T + zero_offset) / (1 + zero_offset) * (np.pi / 2)) ** 2
        alpha_hat = f_t / f_t[0]
        alpha_hat_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_hat[:-1]])
        beta = 1 - alpha_hat / alpha_hat_prev
        return torch.clamp(beta, min=1e-6, max=0.999)

    def noise_data(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        e = torch.randn_like(x).to(self.device)
        return torch.sqrt(self.alpha_hat[t]) * x + torch.sqrt(1 - self.alpha_hat[t]) * e, e
    
    def get_sampling_timesteps(self, n: int) -> Tensor:
        return torch.randint(0, self.T, (n,)).to(self.device)
    
    def train(self, epochs: int, data_loader: DataLoader, loss_function: Callable, optimizer: Optimizer, lr_scheduler: _LRScheduler = None, gradient_accum: int = 1, checkpoint_freq: int = 0, model_path: str = "") -> list[float]:
        logger.info(f"Training started on {self.device}")
        if self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler() #This is obsolete and the other version would work for cuda aswell, but paperspace does not support the other version yet
        else:
            self.scaler = torch.amp.GradScaler(device=self.device)
        loss_list = []
        total_time = 0.0
        self.model.train()

        for e in range(epochs):
            total_loss = 0
            start_time = time.time()
            optimizer.zero_grad()

            for b_idx, (x, _) in enumerate(data_loader):
                x = x.to(self.device).unsqueeze(1)
                timesteps = self.get_sampling_timesteps(x.shape[0])
                x_t, noise = self.noise_data(x, timesteps)
                
                with torch.autocast(device_type=self.device):
                    pred_noise = self.model(x_t, timesteps)
                    loss = loss_function(pred_noise, noise) / gradient_accum

                self.scaler.scale(loss).backward()
                total_loss += loss.item() * gradient_accum

                if (b_idx + 1) % gradient_accum == 0 or (b_idx + 1) == len(data_loader):
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

                if logger.getEffectiveLevel() == LIGHT_DEBUG:
                    print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Batch {b_idx + 1:02d}/{len(data_loader):02d}", end='', flush=True)

            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(flush=True)

            if lr_scheduler is not None:
                lr_scheduler.step()

            avg_loss = total_loss / len(data_loader)
            loss_list.append(avg_loss)

            epoch_time = time.time() - start_time
            total_time += epoch_time
            remaining_time = int((total_time / (e + 1)) * (epochs - e - 1))

            logger.info(f"Epoch {e + 1:02d}: Avg. Loss: {avg_loss:.5e} Remaining Time: {remaining_time // 3600:02d}h {(remaining_time % 3600) // 60:02d}min {round(remaining_time % 60):02d}s LR: {optimizer.param_groups[0]['lr']:.5e}")
            
            if checkpoint_freq > 0 and (e + 1) % checkpoint_freq == 0:
                checkpoint_path: str = f"{model_path[:-4]}_epoch_{e + 1:03d}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                if e + 1 != checkpoint_freq:
                    last_path: str = f"{model_path[:-4]}_epoch_{(e + 1) - checkpoint_freq:03d}.pth"
                    os.remove(last_path)
                logger.light_debug(f"Checkpoint saved model to {checkpoint_path}")

        torch.save(self.model.state_dict(), model_path)
        logger.light_debug(f"Saved model to {model_path}")
        return loss_list
    
    def bwd_diffusion(self, n_samples: int = 8) -> ndarray:
        logger.info(f"Started sampling {n_samples} samples")
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((n_samples, self.inp_dim[-3], self.inp_dim[-2], self.inp_dim[-1])).to(self.device)
            
            for i in reversed(range(1, self.T)):
                if logger.getEffectiveLevel() == LIGHT_DEBUG:
                    print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Sampling timestep {self.T - i}/{self.T}", end='', flush=True)
                
                t = torch.full((n_samples,), i, dtype=torch.long, device=self.device)
                pred_noise = self.model(x, t)

                if i % 10 == 0 or i == 0:
                    print(f"Step {i}:")
                    print(f"  alpha_t: {alpha_t.mean().item():.6f}")
                    print(f"  alpha_hat_t: {alpha_hat_t.mean().item():.6f}")
                    print(f"  beta_t: {beta_t.mean().item():.6f}")
                    print(f"  pred_noise mean/std: {pred_noise.mean().item():.3f}, {pred_noise.std().item():.3f}")
                    print(f"  pred_noise min/max: {pred_noise.min().item():.3f}, {pred_noise.max().item():.3f}")
                    print(f"  x min/max: {x.min().item():.3f}, {x.max().item():.3f}")
                alpha_t = self.alpha.index_select(0, t).view(n_samples, 1, 1, 1)
                alpha_hat_t = self.alpha_hat.index_select(0, t).view(n_samples, 1, 1, 1)
                beta_t = self.beta.index_select(0, t).view(n_samples, 1, 1, 1)

                x = (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise) / torch.sqrt(alpha_t)
                
                if i > 1:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
                
                del pred_noise, noise, t
                torch.cuda.empty_cache()

        if logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(flush=True)
        logger.info(f"Created {n_samples} samples")
        
        self.model.train()
        return x.cpu().numpy()

    
    def visualize_diffusion_steps(self, x: Tensor, n_spectograms: int = 5) -> None:
        x = x.to(self.device).unsqueeze(1)
        batch_size = x.shape[0]

        step_size = self.T // n_spectograms
        selected_timesteps = [i * step_size for i in range(n_spectograms)]

        for t in selected_timesteps:
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            x_t, _ = self.noise_data(x, timesteps)
            
            logger.info(f"Visualizing spectrogram at timestep t={t}")
            visualize_spectogram(x_t[0, 0].cpu().numpy())
