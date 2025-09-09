import torch
from torch import nn, Tensor
from numpy import ndarray
import logging, math
from .Utils import *

logger = logging.getLogger(__name__)


class Diffusion():
    def __init__(self, noise_steps: int, schedule: str = "cosine", inp_shape: list = [12, 1, 262144], device: str = "cpu", filter: Filterbank | PQMF | None = None) -> None:
        """Diffusion Class containing all the functions necessary for diffusion models.

        Args:
            noise_steps (int): The number of noise steps.
            schedule (str, optional): The noise schedule to use is ignored if v-objective diffusion is used. Defaults to "cosine".
            inp_shape (list, optional): The input shape of the audio data. Defaults to [12, 1, 262144].
            device (str, optional): The device to run the model on. Defaults to "cpu".
            filter (Filterbank | PQMF | None, optional): The filterbank to use if diffusion should run on different frequency bands. Defaults to None.
        """
        self.T = noise_steps
        self.fb = filter
        self.device = device
        self.inp_shape = inp_shape
        self.beta = self.get_noise_schedule(schedule).to(self.device)[:, None, None]
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def linear_noise_schedule(self, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
        """Create a linear noise schedule.

        Args:
            beta_start (float, optional): The starting value of the noise schedule. Defaults to 1e-4.
            beta_end (float, optional): The ending value of the noise schedule. Defaults to 2e-2.

        Returns:
            Tensor: The linear noise schedule.
        """
        return torch.linspace(beta_start, beta_end, self.T)
    
    def cos_f(self, t: int, s: float, e: float) -> float:
        """Compute the cosine function for the noise schedule.

        Args:
            t (int): The current timestep.
            s (float): The starting value of the noise schedule.
            e (float): The ending value of the noise schedule.

        Returns:
            float: The computed cosine value.
        """
        return math.cos(((t / self.T + s) / (1 + s)) * (math.pi / 2)) ** e
    
    def cosine_noise_schedule(self, s: float = 8e-3, e: float = 2) -> Tensor:
        """Create a cosine noise schedule.

        Args:
            s (float, optional): The starting value of the noise schedule. Defaults to 8e-3.
            e (float, optional): The ending value of the noise schedule. Defaults to 2.

        Returns:
            Tensor: The cosine noise schedule.
        """
        f_t = [self.cos_f(t, s, e) for t in range(self.T)]
        alpha_hat_t = [f / f_t[0] for f in f_t]
        alpha = [alpha_hat_t[0]] + [alpha_hat_t[i] / alpha_hat_t[i - 1] for i in range(1, self.T)]
        return 1 - Tensor(alpha)
    
    def get_noise_schedule(self, schedule: str) -> Tensor:
        """Get the noise schedule.

        Args:
            schedule (str): The noise schedule to use.

        Returns:
            Tensor: The noise schedule.
        """
        if schedule == "cosine":
            return self.cosine_noise_schedule()
        if schedule == "linear":
            return self.linear_noise_schedule()
        else:
            logger.fatal(f"Invalid Noise Schedule {schedule}")
    
    def noise_data(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Add noise to the input data.

        Args:
            x (Tensor): The input data.
            t (Tensor): The current timestep.

        Returns:
            tuple[Tensor, Tensor]: The noisy input data and the noise added.
        """
        e = torch.randn_like(x).to(self.device)
        return torch.sqrt(self.alpha_hat[t]) * x + torch.sqrt(1 - self.alpha_hat[t]) * e, e
    
    def get_sampling_timesteps(self, n: int) -> Tensor:
        """Get the sampling timesteps.

        Args:
            n (int): The number of timesteps to sample.

        Returns:
            Tensor: The sampled timesteps.
        """
        return torch.randint(0, self.T, (n,)).to(self.device).to(torch.long)
    
    def prep_train_ddxm(self, inp: Tensor) -> tuple[Tensor, ...]:
        """Prepare the input data for training the DDIM or DDPM model applies the filterbank if available.

        Args:
            inp (Tensor): The input data.

        Returns:
            tuple[Tensor, ...]: The prepared input data.
        """
        if self.fb:
            inp = self.fb.analysis(inp)
        timesteps = self.get_sampling_timesteps(inp.shape[0])
        x_t, noise = self.noise_data(inp, timesteps)
        return x_t, noise, timesteps
    
    def x0_from_v(self, v: Tensor, x_sigma: Tensor, sigma_b: Tensor) -> Tensor:
        """Compute x0 from v, x_sigma, and sigma_b.

        Args:
            v (Tensor): The input tensor v.
            x_sigma (Tensor): The input tensor x_sigma.
            sigma_b (Tensor): The input tensor sigma_b.

        Returns:
            Tensor: The computed x0 tensor.
        """
        a, b = self.get_semicircle_weights(sigma_b)
        eps_hat = (v + b * x_sigma) / a
        x0_hat  = (a * eps_hat - v) / b
        if self.fb:
            x0_hat = self.fb.synthesis(x0_hat, self.inp_shape[-1])
        return x0_hat

    def get_semicircle_weights(self, sigma_t: Tensor) -> tuple[Tensor, ...]:
        """Get the semicircle weights for v-diffusion.

        Args:
            sigma_t (Tensor): The timestep tensor sigma_t.

        Returns:
            tuple[Tensor, ...]: The semicircle weights.
        """
        phi_t = (torch.pi / 2.0) * sigma_t
        alpha = torch.cos(phi_t)
        beta = torch.sin(phi_t)
        return alpha.view(-1, 1, 1), beta.view(-1, 1, 1)
    
    def noise_img_v_obj(self, x_0: Tensor, sigma_t: Tensor) -> tuple[Tensor, ...]:
        """Add noise to the input image for v-diffusion.

        Args:
            x_0 (Tensor): The original image tensor.
            sigma_t (Tensor): The timestep tensor sigma_t.

        Returns:
            tuple[Tensor, ...]: The noisy image tensor and the noise tensor.
        """
        alpha, beta = self.get_semicircle_weights(sigma_t)
        epsilon = torch.randn_like(x_0).to(self.device)
        x_sigma_t = alpha * x_0 + beta * epsilon
        return x_sigma_t, epsilon
    
    def prep_train_v_obj(self, inp: Tensor) -> tuple[Tensor, ...]:
        """Prepare the input data for training the v-diffusion model.

        Args:
            inp (Tensor): The input data.

        Returns:
            tuple[Tensor, ...]: The prepared input data.
        """
        if self.fb:
            inp = self.fb.analysis(inp)
        sigma_t = torch.rand(inp.shape[0]).to(self.device)
        x_sigma_t, e = self.noise_img_v_obj(inp, sigma_t)
        a, b = self.get_semicircle_weights(sigma_t)
        true_vel = a * e - b * inp
        return true_vel, x_sigma_t, sigma_t

    def bwd_diffusion_ddpm(self, model: nn.Module, shape: list, seed: Tensor | None = None) -> ndarray:
        """Backward diffusion process for DDPM.

        Args:
            model (nn.Module): The diffusion model.
            shape (list): The shape of the input tensor.
            seed (Tensor | None, optional): The seed tensor for initialization. Defaults to None.

        Returns:
            ndarray: The generated samples.
        """
        logger.info(f"Started sampling {shape[0]} samples on {self.device}")
        
        model.eval()

        timesteps = self.T
        n_dim = len(shape)
        batch = shape[0]
        
        x = torch.randn(shape).to(self.device) if seed is None else seed
        if self.fb:
                    x = self.fb.analysis(x)

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
        if self.fb:
            return self.fb.synthesis(x, shape[-1]).cpu().numpy(), x.cpu().numpy()
        return x.cpu().numpy()
    
    def bwd_diffusion_ddim(self, model: nn.Module, shape: list, n_steps: int, eta: float = 0.0, seed: Tensor | None = None) -> ndarray:
        """Backward diffusion process for DDIM.

        Args:
            model (nn.Module): The diffusion model.
            shape (list): The shape of the input tensor.
            n_steps (int): The number of diffusion steps.
            eta (float, optional): The noise schedule. Defaults to 0.0.
            seed (Tensor | None, optional): The seed tensor for initialization. Defaults to None.

        Returns:
            ndarray: The generated samples.
        """
        logger.info(f"Started sampling {shape[0]} samples on {self.device}")
        model.eval()

        timesteps = self.T
        batch = shape[0]

        timesteps_ind = torch.linspace(0, timesteps - 1, steps=n_steps, dtype=torch.long, device=self.device)

        x = torch.randn(shape).to(self.device) if seed is None else seed
        if self.fb:
            x = self.fb.analysis(x)
        
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
        if self.fb:
            return self.fb.synthesis(x, shape[-1]).cpu().numpy(), x.cpu().numpy()
        return x.cpu().numpy()
    
    def bwd_diffusion_v_obj(self, model: nn.Module, shape: list, n_steps: int, seed: Tensor | None = None, seed_fwd_steps: int = 0) -> ndarray:
        """Backward diffusion process for v-diffusion.

        Args:
            model (nn.Module): The diffusion model.
            shape (list): The shape of the input tensor.
            n_steps (int): The number of diffusion steps.
            seed (Tensor | None, optional): The seed tensor for initialization. Defaults to None.

        Returns:
            ndarray: The generated samples.
        """
        logger.info(f"Started sampling {shape[0]} samples on {self.device}")
        model.eval()

        batch = shape[0]
        sigmas = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
        start_step = 0
        if seed is not None:
            a, b = self.get_semicircle_weights(sigmas[n_steps - seed_fwd_steps])
            x = seed * a + b * torch.randn_like(seed)
            start_step = n_steps - seed_fwd_steps
        else:
            if self.fb:
                x = torch.randn((batch, self.fb.N, shape[-1]), device=self.device) 
            else:
                x = torch.randn(shape, device=self.device)

        for i in range(start_step, n_steps):
            sigma_t = sigmas[i]
            sigma_tp1 = sigmas[i+1]

            sigma_t_b = torch.full((batch,), sigma_t, device=self.device)
            sigma_tp1_b = torch.full((batch,), sigma_tp1, device=self.device)

            with torch.no_grad():
                v_pred = model(x, sigma_t_b)

            a, b = self.get_semicircle_weights(sigma_t_b)
            a1, b1 = self.get_semicircle_weights(sigma_tp1_b)

            x_pred = a * x - b * v_pred
            noise_pred = b * x + a * v_pred
            x = a1 * x_pred + b1 * noise_pred

        logger.info(f"Created {batch} samples")
        if self.fb:
            return self.fb.synthesis(x, shape[-1]).cpu().numpy(), x.cpu().numpy()
        return x.cpu().numpy()
