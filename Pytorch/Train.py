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

batch_size: int = 1
epochs: int = 300
learning_rate: float = 1e-4
lr_decay: int = 40
lr_gamma: float = 0.1
reprod_loss_weight: float = 20000
logging_level: int = logging.INFO
model_name: str = "audio_vae_v4_mod"
training_data_name: str = "training_v2"


logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

file = load_training_data(f"{DATA_PATH}/{training_data_name}.npy")[40, ...].reshape(1, 1024,672)
data_loader = create_dataloader(Audio_Data(file), batch_size)
logger.info(f"Data loaded with shape: {file.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(in_channels=1, latent_dim=4096, device=device,input_shape=[0,0, file.shape[-2], file.shape[-1]], n_conv_blocks=1, n_starting_filters=16).to(device)
if os.path.exists(f"{MODEL_PATH}/{model_name}.pth"):
    model.load_state_dict(torch.load(f"{MODEL_PATH}/{model_name}.pth"))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_gamma)
x = train_VAE(model, data_loader, optimizer, loss_VAE, epochs=epochs, device=device, reprod_loss_weight=reprod_loss_weight)
scatter_plot(x)
torch.save(model.state_dict(), f"{MODEL_PATH}/{model_name}.pth")
logger.info("Model saved successfully.")