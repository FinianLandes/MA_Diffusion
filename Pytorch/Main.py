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
# Base Scripts
from VAE import *
from Utils import *



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)
data_path: str = "Data"

sample_rate: int = 16000
batch_size: int = 40
epochs: int = 100

filenames = get_filenames_from_folder(data_path, "wav")
file = load_audio_file(os.path.join(data_path, filenames[1]), sample_rate, True, True)
file = split_audiofile(file, 3, sample_rate)[:2000, :]
data_loader = create_dataloader(Audio_Data(file), batch_size)
x_dim = file.shape[1]
logger.info(f"Data loaded with shape: {file.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(x_dim, 2048, 1024, device, nn.Sigmoid()).to(device)

#model.load_state_dict(torch.load("Models/audio_vae_v1.pth"))
#model.eval()

optimizer = optim.Adam(model.parameters(), lr =5e-6)
#samples = generate_sample(model, device, data_loader.dataset[10][0]).flatten()
#print(np.min(samples), np.max(samples), samples.shape)
#save_file(samples, "test.wav", sample_rate)

train_VAE(model, data_loader, optimizer, loss_VAE, batch_size, x_dim, epochs, device, logger)
torch.save(model.state_dict(), "Models/audio_vae_v1.pth")
