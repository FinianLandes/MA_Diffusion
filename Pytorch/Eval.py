# Torch
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
# Utils
import numpy as np
from numpy import ndarray
import logging, os
# Base Scripts
from VAE import *
from Utils import *
from Conf import *

logging_level: int = LIGHT_DEBUG
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

def generate_data(model: nn.Module, n_samples: int = 1, seed: ndarray = None, file_name: str = "test") -> None:
    model.eval()
    if seed is not None:
        save_audio_file(spectrogram_to_audio(seed, LEN_FFT), f"{RESULT_PATH}/{file_name}_seed.wav")
        samples = generate_sample(model, device, Tensor(seed).view(1, 1, seed.shape[-2], seed.shape[-1]), num_samples=n_samples)
    else:
        samples = generate_sample(model, device, num_samples=n_samples)
    for i in range(samples.shape[1]):
        save_audio_file(spectrogram_to_audio(samples[0,i], LEN_FFT), f"{RESULT_PATH}/{file_name}_{i:02d}.wav", SAMPLE_RATE)
    logger.light_debug(f"Saved {samples.shape[0]} samples to {RESULT_PATH}")

def pass_through(sample: ndarray, file_name: str = "test") -> None:
    save_audio_file(spectrogram_to_audio(file[100], LEN_FFT), f"{RESULT_PATH}/{file_name}_inp.wav")
    x = fwd_pass(model, device, Tensor(sample).view(1, 1, sample.shape[-2], sample.shape[-1]))
    save_audio_file(spectrogram_to_audio(x[0,0], LEN_FFT), f"{RESULT_PATH}/{file_name}_out.wav")
    logger.light_debug(f"Saved passed through sample to {RESULT_PATH}")

file = load_training_data(DATA_PATH+"/training_v1.npy")
logger.info(f"Data loaded with shape: {file.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(in_channels=1, latent_dim=512, device=device,input_shape=[0,0, file.shape[-2], file.shape[-1]]).to(device)
model.load_state_dict(torch.load("Models/audio_vae_v1.pth"))

#generate_data(model=model, seed=file[100])
pass_through(file[70])
