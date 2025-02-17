# Torch
import torch
from torch.utils.data import DataLoader, Dataset
# Utils
import numpy as np
from numpy import ndarray
import logging, os
# Base Scripts
from Utils import *
from Conf import *

logging_level: int = LIGHT_DEBUG
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

training_data_name: str = "training_v2"
overlap: int = 3
filenames = get_filenames_from_folder(DATA_PATH, "wav")
data: list = []
for i in range(len(filenames)):
    file = load_audio_file(os.path.join(DATA_PATH, filenames[i]), SAMPLE_RATE, True)
    file = split_audiofile(file, TIME_FRAME_S, SAMPLE_RATE, overlap)
    file = audio_splits_to_spectograms(file, LEN_FFT)
    file = normalize(file)
    data.append(file)
data: ndarray = np.vstack(data)
data = dimension_for_VAE(data)
logger.info(f"Processed data of shape: {data.shape}")
save_training_data(data, f"{DATA_PATH}/{training_data_name}.npy")