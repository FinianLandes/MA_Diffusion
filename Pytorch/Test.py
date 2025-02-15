# Torch
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
# Utils
import numpy as np
from numpy import ndarray
import logging
import os
from torch_lr_finder import LRFinder
# Base Scripts
from VAE import *
from Utils import *

sample_rate: int = 44100
batch_size: int = 40
epochs: int = 100
data_path: str = "Data"
logging_level: int = logging.INFO

logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.    Logger = logging.getLogger(__name__)



filenames = get_filenames_from_folder(data_path, "wav")
file = load_audio_file(os.path.join(data_path, filenames[1]), sample_rate, True)
file = split_audiofile(file, 3, sample_rate)[:10, :]
file = audio_splits_to_spectograms(file, 2048)[:, :1024, :256]
data_loader = create_dataloader(Audio_Data(file), batch_size)
logger.info(f"Data loaded with shape: {file.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(in_channels=1, latent_dim=512, device=device,input_shape=[0,0, 1024, 256]).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_VAE(model, data_loader, optimizer, loss_VAE, epochs=epochs, device=device)

torch.save(model.state_dict(), "Models/audio_vae_v1.pth")
logger.info("Model saved successfully.")