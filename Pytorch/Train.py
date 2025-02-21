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

batch_size: int = 16
epochs: int = 100
learning_rate: float = 5e-4
lr_decay: int = 40
lr_gamma: float = 0.1
reprod_loss_weight: float = 20000
logging_level: int = LIGHT_DEBUG #logging.INFO
model_name: str = "conv_VAE_v1"
model_path: str = f"{MODEL_PATH}/{model_name}.pth"
checkpoint_freq: int = 5 #0 for no checkpoint saving
training_data_name: str = "training_v2"


logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

file = load_training_data(f"{DATA_PATH}/{training_data_name}.npy")[:16*5, ...]
data_loader = create_dataloader(Audio_Data(file), batch_size)
logger.info(f"Data loaded with shape: {file.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(in_channels=1, latent_dim=128, device=device,input_shape=[0,0, file.shape[-2], file.shape[-1]], n_conv_blocks=1, n_starting_filters=32, lin_bottleneck=False).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    logger.info(f"Model {model_name} loaded with {count_parameters(model)} Parameters")
else: 
    logger.info(f"Model {model_name} created with {count_parameters(model)} Parameters")
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_gamma)
x = train_VAE(model, data_loader, optimizer, loss_VAE, epochs=epochs, device=device, reprod_loss_weight=reprod_loss_weight, checkpoint_freq=checkpoint_freq, model_path=model_path)
scatter_plot(x)
torch.save(model.state_dict(), model_path)
logger.info("Model saved successfully.")