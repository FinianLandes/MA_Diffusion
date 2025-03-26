import torch
from torch import nn, Tensor, functional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
import numpy as np
from numpy import ndarray
import logging, time, math
from typing import Callable
from .Utils import *
from .U_Net import *
from MainScripts.Conf import conf

logger = logging.getLogger(__name__)

class Diffusion():
    """The DDPM & DDIM class.
    """
    def __init__(self, model: nn.Module, noise_steps: int, noise_schedule: str = "linear", input_dim: list = [1, 1, 1024, 672], device: str = "cpu") -> None:
        """Initializer for the DDPM class.

        Args:
            model (nn.Module): The neural net to learn the noise distribution takes to paramters: the noised image (B, C, H, W) and the time embeding (B,).
            noise_steps (int): Number of noising steps.
            noise_schedule (str, optional): The noise schedule type. Defaults to "cosine".
            input_dim (list, optional): The input shape in format (B, C, H, W). Defaults to [1, 1, 1024, 672].
            device (str, optional): The training device. Defaults to "cpu".
        """
        self.model = model
        self.T = noise_steps
        self.inp_dim = input_dim
        self.device = device
        self.beta = self.get_noise_schedule(noise_schedule).to(self.device)[:, None, None, None]
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def get_noise_schedule(self, noise_type: str = "linear") -> Tensor:
        """Gets a noise schedule.

        Args:
            noise_type (str, optional): The noise type. Defaults to "linear".

        Returns:
            Tensor: Returns a noise schedule as beta values.
        """
        if noise_type == "cosine":
            return self.cosine_noise_schedule()
        if noise_type == "linear":
            return self.linear_noise_schedule()
        else:
            logger.fatal(f"Invalid Noise Schedule {noise_type}")
    
    def linear_noise_schedule(self, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
        """Creates a linear noise schedule

        Args:
            beta_start (float, optional): Minimum beta value. Defaults to 1e-4.
            beta_end (float, optional): Maximal beta value. Defaults to 2e-2.

        Returns:
            Tensor: The noise schedule as beta values.
        """
        return torch.linspace(beta_start, beta_end, self.T)
    
    def cos_f(self, t: int, s: float, e: float) -> float:
        return math.cos(((t / self.T + s) / (1 + s)) * (math.pi / 2)) ** e
    def cosine_noise_schedule(self, s: float = 8e-3, e: float = 2) -> Tensor:
        f_t = [self.cos_f(t, s, e) for t in range(self.T)]
        alpha_hat_t = [f / f_t[0] for f in f_t]
        alpha = [alpha_hat_t[0]] + [alpha_hat_t[i] / alpha_hat_t[i - 1] for i in range(1, self.T)]
        return 1 - Tensor(alpha)

    def noise_data(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Adds noise to the data according to the noise schedule and the timesteps.

        Args:
            x (Tensor): The data to noise.
            t (Tensor): the timesteps.

        Returns:
            tuple[Tensor, Tensor]: The noised data and the noise.
        """
        e = torch.randn_like(x).to(self.device)
        return torch.sqrt(self.alpha_hat[t]) * x + torch.sqrt(1 - self.alpha_hat[t]) * e, e
    
    def get_sampling_timesteps(self, n: int) -> Tensor:
        """Get n random timesteps

        Args:
            n (int): Number of random timesteps.

        Returns:
            Tensor: Tensor of shape (n,).
        """
        return torch.randint(0, self.T, (n,)).to(self.device)
    
    def train(self, epochs: int, data_loader: DataLoader, loss_function: Callable, optimizer: Optimizer, lr_scheduler: _LRScheduler = None, gradient_accum: int = 1, checkpoint_freq: int = 0, model_path: str = "test_model", start_epoch: int = 0, patience: int = 20) -> list[float]:
        """Training of the diffusion model.

        Args:
            epochs (int): number of epochs to train
            data_loader (DataLoader): Dataloader containing the training data.
            loss_function (Callable): The loss function.
            optimizer (Optimizer): An optimizer from the .optim class.
            lr_scheduler (_LRScheduler, optional): An lr scheduler. Defaults to None.
            gradient_accum (int, optional): If >1 accumulates the gradients over the given number of batches. Defaults to 1.
            checkpoint_freq (int, optional): If >0 saves the model each n epochs. Defaults to 0.
            model_path (str, optional): The model path to save the model to. Defaults to "test_model".
            start_epoch (int, optional): The starting epoch (just for logging reasons). Defaults to 0.
            patience (int, optional): If > 0 stops training if loss does not improve after given number of epochs. Defaults to 20.

        Returns:
            list[float]: The epochs avg. losses as a list.
        """

        logger.info(f"Training started on {self.device}")
        if self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler() #This is obsolete and the other version would work for cuda aswell, but paperspace does not support the other version yet
        else:
            self.scaler = torch.amp.GradScaler(device=self.device)
        loss_list: list = []
        total_time: float = 0.0
        best_loss = float('inf')
        epochs_no_improve: int = 0

        self.model.train()
        for e in range(0, epochs):
            total_loss: float = 0
            min_noise = [0.0, 0.0]
            max_noise = [0.0, 0.0]
            std_noise = [0.0, 0.0]
            mean_noise = [0.0, 0.0]
            start_time: float = time.time()
            optimizer.zero_grad()

            for b_idx, (x, _) in enumerate(data_loader):
                if x.dim() == 3:
                    x = x.to(self.device).unsqueeze(1)
                else:
                    x = x.to(self.device)
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
                    min_noise[0] = min(torch.min(pred_noise).item(), min_noise[0])
                    min_noise[1] = min(torch.min(noise).item(), min_noise[1])
                    max_noise[0] = max(torch.max(pred_noise).item(), max_noise[0])
                    max_noise[1] = max(torch.max(noise).item(), max_noise[1])
                    std_noise[0] += torch.std(pred_noise).item()
                    std_noise[1] += torch.std(noise).item()
                    mean_noise[0] += torch.mean(pred_noise).item()
                    mean_noise[1] += torch.mean(noise).item()
                    current_batch = b_idx + 1
                    print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Batch {current_batch:02d}/{len(data_loader):02d} Pred noise min/max: {min_noise[0]:.5f}, {max_noise[0]:.5f} std/mean: {std_noise[0] / current_batch:.5f}, {mean_noise[0] / current_batch:.5f} Real noise min/max: {min_noise[1]:.5f}, {max_noise[1]:.5f} std/mean: {std_noise[1] / current_batch:.5f}, {mean_noise[1] / current_batch:.5f} ", end='', flush=True)

            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(flush=True)

            avg_loss = total_loss / len(data_loader)
            loss_list.append(avg_loss)

            if lr_scheduler is not None:
                if isinstance(lr_scheduler, (optim.lr_scheduler.ReduceLROnPlateau, Threshold_LR)):
                    lr_scheduler.step(avg_loss)
                else:
                    lr_scheduler.step()

            if patience > 0:
                if avg_loss < best_loss:
                    epochs_no_improve = 0
                    best_loss = avg_loss
                else:
                    epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {e + 1}: Loss has not improved for {patience} epochs")
                break

            epoch_time = time.time() - start_time
            total_time += epoch_time
            remaining_time = int((total_time / (e + 1)) * (epochs - e - 1))

            logger.info(f"Epoch {e + 1 + start_epoch:03d}: Avg. Loss: {avg_loss:.5e} Remaining Time: {remaining_time // 3600:02d}h {(remaining_time % 3600) // 60:02d}min {round(remaining_time % 60):02d}s LR: {optimizer.param_groups[0]['lr']:.5e} ")
            
            if checkpoint_freq > 0 and (e + 1) % checkpoint_freq == 0:
                checkpoint_path: str = f"{model_path[:-4]}_epoch_{e + 1:03d}.pth"
                torch.save({"model": self.model.state_dict(), "optim": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(), "epoch": e + 1}, checkpoint_path)
                if e + 1 != checkpoint_freq:
                    last_path: str = f"{model_path[:-4]}_epoch_{(e + 1) - checkpoint_freq:03d}.pth"
                    del_if_exists(last_path)
                logger.light_debug(f"Checkpoint saved model to {checkpoint_path}")

        torch.save({"model": self.model.state_dict(), "optim": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(), "epoch": e + 1}, model_path)
        logger.light_debug(f"Saved model to {model_path}")

        if checkpoint_freq > 0:
            checkpoint_path: str = f"{model_path[:-4]}_epoch_{e + 1 - ((e + 1) % checkpoint_freq):03d}.pth"
            del_if_exists(checkpoint_path)
        
        return loss_list
    
    def bwd_diffusion_ddpm(self, n_samples: int = 8, visual_freq: int = 0) -> ndarray:
        """The bwd diffusion process with DDPM.

        Args:
            n_samples (int, optional): Number of samples to generate. If model contains batch norm creating >1 sample is more efficient. Defaults to 8.
            visual_freq (int, optional): If >0 visualizes the spectogram each n steps. Defaults to 0.

        Returns:
            ndarray: The generated samples.
        """
        logger.info(f"Started sampling {n_samples} samples")
        self.model.eval()
        timesteps = self.T

        x = torch.randn((n_samples, self.inp_dim[-3], self.inp_dim[-2], self.inp_dim[-1])).to(self.device)
        for i in reversed(range(1, timesteps)):
            x = x / (x.std() + 1e-8)

            t = torch.full((n_samples,), i, dtype=torch.long, device=self.device)
            with torch.no_grad():
                pred_noise = self.model(x, t)

            alpha_t = self.alpha.index_select(0, t).view(n_samples, 1, 1, 1)
            alpha_hat_t = self.alpha_hat.index_select(0, t).view(n_samples, 1, 1, 1)
            beta_t = self.beta.index_select(0, t).view(n_samples, 1, 1, 1)

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise)
            
            if i > 1:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
            
            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Sampling timestep {timesteps - i}/{timesteps} X min/max: {torch.min(x).item():.5f}, {torch.max(x).item():.5f} noise min/max: {torch.min(pred_noise).item():.5f}, {torch.max(pred_noise).item():.5f} std/mean: {torch.std(pred_noise).item():.5f}, {torch.mean(pred_noise).item():.5f} ", end='', flush=True)
            
            if visual_freq > 0 and i % visual_freq == 0:
                visualize_spectogram(normalize(x.cpu().numpy()[0, 0], -1, 1),conf["audio"].sample_rate, conf["audio"].len_fft )


        if logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(flush=True)
        logger.light_debug(f"Final X min/max before return: {torch.min(x).item():.5f}, {torch.max(x).item():.5f}")
        logger.info(f"Created {n_samples} samples")

        self.model.train()
        return x.cpu().numpy()

    def bwd_diffusion_ddim(self, n_samples: int = 8, sampling_timesteps: int = 50, eta: float = 0.0, visual_freq: int = 0) -> ndarray:
        """The bwd diffusion process as DDIM.

        Args:
            n_samples (int, optional): Number of samples to generate. If model contains batch norm creating >1 sample is more efficient. Defaults to 8.
            sampling_timesteps (int, optional): Number of actual sampling timesteps. Defaults to 50.
            eta: Stochasticity parameter (0.0 for deterministic, 1.0 for DDPM-like stochasticity). Defaults to 0. 
            visual_freq (int, optional): If >0 visualizes the spectogram each n steps. Defaults to 0.

        Returns:
            ndarray: The generated samples.
        """
        logger.info(f"Started sampling {n_samples} samples")
        self.model.eval()
        timesteps = self.T
        
        timesteps_ind = torch.linspace(0, timesteps - 1, steps=sampling_timesteps, dtype=torch.long, device=self.device)

        x = torch.randn((n_samples, self.inp_dim[-3], self.inp_dim[-2], self.inp_dim[-1])).to(self.device)
        
        for i in reversed(range(1, sampling_timesteps)):
            t = timesteps_ind[i]
            t_prev = timesteps_ind[i - 1]

            alpha_hat_t = self.alpha_hat[t]
            alpha_hat_prev = self.alpha_hat[t_prev]
            t = torch.full((n_samples,), i, dtype=torch.long, device=self.device)
            with torch.no_grad():
                pred_noise = self.model(x, t)
            
            sigma_t = eta * torch.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat_t)) * torch.sqrt(1 - alpha_hat_t / alpha_hat_prev)

            x0_pred = (x - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
            x = torch.sqrt(alpha_hat_prev) * x0_pred + torch.sqrt(1 - alpha_hat_prev - sigma_t**2) * pred_noise

            if eta > 0 and i > 1:
                x += sigma_t * torch.randn_like(x)
            
            if logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Sampling timestep {sampling_timesteps - i}/{sampling_timesteps} X min/max: {torch.min(x).item():.5f}, {torch.max(x).item():.5f} noise min/max: {torch.min(pred_noise).item():.5f}, {torch.max(pred_noise).item():.5f} std/mean: {torch.std(pred_noise).item():.5f}, {torch.mean(pred_noise).item():.5f} ", end='', flush=True)
            
            if visual_freq > 0 and i % visual_freq == 0:
                visualize_spectogram(normalize(x.cpu().numpy()[0, 0], -1, 1),conf["audio"].sample_rate, conf["audio"].len_fft )

        if logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(flush=True)
        logger.info(f"Created {n_samples} samples")

        self.model.train()
        return x.cpu().numpy()
    
    def visualize_diffusion_steps(self, x: Tensor, n_spectograms: int = 5) -> None:
        """Visualizes the noise schedule applied to a spectogram.

        Args:
            x (Tensor): The spectograms to show if more than one are provided first one is shown.
            n_spectograms (int, optional): Number of steps to visualize. Defaults to 5.
        """
        if x.dim() == 3:
            x = x.to(self.device).unsqueeze(1)
        else:
            x = x.to(self.device)
        batch_size = x.shape[0]

        step_size = self.T // n_spectograms
        selected_timesteps = [i * step_size for i in range(n_spectograms)]

        for t in selected_timesteps:
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            x_t, _ = self.noise_data(x, timesteps)
            
            logger.info(f"Visualizing spectrogram at timestep t={t}")
            visualize_spectogram(x_t[0, 0].cpu().numpy())

