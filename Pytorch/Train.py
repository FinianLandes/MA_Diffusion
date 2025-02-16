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

batch_size: int = 32
epochs: int = 20
learning_rate: float = 1e-6
reprod_loss_weight: float = 1000
logging_level: int = LIGHT_DEBUG


logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

file = load_training_data(DATA_PATH + "/training_v1.npy")[:2080, ...]
data_loader = create_dataloader(Audio_Data(file), batch_size)
logger.info(f"Data loaded with shape: {file.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(in_channels=1, latent_dim=512, device=device,input_shape=[0,0, file.shape[-2], file.shape[-1]]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
train_VAE(model, data_loader, optimizer, loss_VAE, epochs=epochs, device=device, reprod_loss_weight=reprod_loss_weight)

torch.save(model.state_dict(), "Models/audio_vae_v1.pth")
logger.info("Model saved successfully.")