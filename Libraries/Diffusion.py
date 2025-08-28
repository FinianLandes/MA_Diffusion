import torch
from torch import nn, Tensor
import numpy as np
from numpy import ndarray
import logging, math
from .Utils import *

logger = logging.getLogger(__name__)


class Diffusion():
    def __init__(self, noise_steps: int, schedule: str = "cosine", v_obj: bool = False, device: str = "cpu") -> None:
        self.T = noise_steps
        self.device = device
        self.beta = self.get_noise_schedule(schedule).to(self.device)[:, None, None]
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def linear_noise_schedule(self, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
        return torch.linspace(beta_start, beta_end, self.T)
    
    def cos_f(self, t: int, s: float, e: float) -> float:
        return math.cos(((t / self.T + s) / (1 + s)) * (math.pi / 2)) ** e
    
    def cosine_noise_schedule(self, s: float = 8e-3, e: float = 2) -> Tensor:
        f_t = [self.cos_f(t, s, e) for t in range(self.T)]
        alpha_hat_t = [f / f_t[0] for f in f_t]
        alpha = [alpha_hat_t[0]] + [alpha_hat_t[i] / alpha_hat_t[i - 1] for i in range(1, self.T)]
        return 1 - Tensor(alpha)
    
    def get_noise_schedule(self, schedule: str) -> Tensor:
        if schedule == "cosine":
            return self.cosine_noise_schedule()
        if schedule == "linear":
            return self.linear_noise_schedule()
        else:
            logger.fatal(f"Invalid Noise Schedule {schedule}")
    
    def noise_data(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        e = torch.randn_like(x).to(self.device)
        return torch.sqrt(self.alpha_hat[t]) * x + torch.sqrt(1 - self.alpha_hat[t]) * e, e
    
    def get_sampling_timesteps(self, n: int) -> Tensor:
        return torch.randint(0, self.T, (n,)).to(self.device).to(torch.long)
    
    def prep_train_ddxm(self, inp: Tensor) -> tuple[Tensor, ...]:
        timesteps = self.get_sampling_timesteps(inp.shape[0])
        x_t, noise = self.noise_data(inp, timesteps)
        return x_t, noise, timesteps
    
    
    def get_semicircle_weights(self, sigma_t: Tensor) -> tuple[Tensor, ...]:
            phi_t = (torch.pi / 2.0) * sigma_t
            alpha = torch.cos(phi_t)
            beta = torch.sin(phi_t)
            return alpha.view(-1, 1, 1), beta.view(-1, 1, 1)
    
    def noise_img_v_obj(self, x_0: Tensor, sigma_t: Tensor) -> tuple[Tensor, ...]:
        alpha, beta = self.get_semicircle_weights(sigma_t)
        epsilon = torch.randn_like(x_0).to(self.device)
        x_sigma_t = alpha * x_0 + beta * epsilon
        return x_sigma_t, epsilon
    
    def prep_train_v_obj(self, inp: Tensor) -> tuple[Tensor, ...]:
        sigma_t = torch.rand(inp.shape[0]).to(self.device)
        x_sigma_t, e = self.noise_img_v_obj(inp, sigma_t)
        a, b = self.get_semicircle_weights(sigma_t)
        true_vel = a * e - b * inp
        return true_vel, x_sigma_t, sigma_t

    def bwd_diffusion_ddpm(self, model: nn.Module, shape: list, seed: Tensor | None = None) -> ndarray:
        logger.info(f"Started sampling {shape[0]} samples on {self.device}")
        
        model.eval()

        timesteps = self.T
        n_dim = len(shape)
        batch = shape[0]
        
        x = torch.randn(shape).to(self.device) if seed is None else seed
        for i in reversed(range(1, timesteps)):
            t = torch.full((batch,), i, dtype=torch.long, device=self.device)
            with torch.no_grad():
                pred_noise = model(x, t)

            alpha_t = self.alpha.index_select(0, t).view(*[batch] + [1 for _ in range(n_dim - 1)])
            alpha_hat_t = self.alpha_hat.index_select(0, t).view(*[batch] + [1 for _ in range(n_dim - 1)])
            beta_t = self.beta.index_select(0, t).view(*[batch] + [1 for _ in range(n_dim - 1)])

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise)
            
            if i > 1:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        logger.info(f"Created {batch} samples")
        return x.cpu().numpy()
    
    def bwd_diffusion_ddim(self, model: nn.Module, shape: list, n_steps: int, eta: float = 0.0, seed: Tensor | None = None) -> ndarray:
        logger.info(f"Started sampling {shape[0]} samples on {self.device}")
        model.eval()

        timesteps = self.T
        batch = shape[0]

        timesteps_ind = torch.linspace(0, timesteps - 1, steps=n_steps, dtype=torch.long, device=self.device)

        x = torch.randn(shape).to(self.device) if seed is None else seed
        
        for i in reversed(range(1, n_steps)):
            t = timesteps_ind[i]
            t_prev = timesteps_ind[i - 1]

            alpha_hat_t = self.alpha_hat[t]
            alpha_hat_prev = self.alpha_hat[t_prev]
            t = torch.full((batch,), i, dtype=torch.long, device=self.device)
            with torch.no_grad():
                pred_noise = model(x, t)
            
            sigma_t = eta * torch.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat_t)) * torch.sqrt(1 - alpha_hat_t / alpha_hat_prev)

            x0_pred = (x - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
            x = torch.sqrt(alpha_hat_prev) * x0_pred + torch.sqrt(1 - alpha_hat_prev - sigma_t**2) * pred_noise

            if eta > 0 and i > 1:
                x += sigma_t * torch.randn_like(x)
        
        logger.info(f"Created {batch} samples")
        return x.cpu().numpy()
    
    def bwd_diffusion_v_obj(self, model: nn.Module, shape: list, n_steps: int, seed: Tensor | None = None):
        logger.info(f"Started sampling {shape[0]} samples on {self.device}")
        model.eval()

        batch = shape[0]
        x = torch.randn(shape, device=self.device) if seed is None else seed.to(self.device)

        # reverse sigmas: 1 → 0
        sigmas = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)

        for i in range(n_steps):
            sigma_t   = sigmas[i]
            sigma_tp1 = sigmas[i+1]

            sigma_t_b   = torch.full((batch,), sigma_t,   device=self.device)
            sigma_tp1_b = torch.full((batch,), sigma_tp1, device=self.device)

            with torch.no_grad():
                v_pred = model(x, sigma_t_b)

            # weights
            a, b   = self.get_semicircle_weights(sigma_t_b)
            a1, b1 = self.get_semicircle_weights(sigma_tp1_b)

            # Archisound-style update (no divisions!)
            x_pred    = a * x - b * v_pred
            noise_pred = b * x + a * v_pred
            x = a1 * x_pred + b1 * noise_pred

        logger.info(f"Created {batch} samples")
        return x.cpu().numpy()
