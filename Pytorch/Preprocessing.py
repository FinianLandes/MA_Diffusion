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

filenames = get_filenames_from_folder(DATA_PATH, "wav")
file = load_audio_file(os.path.join(DATA_PATH, filenames[1]), SAMPLE_RATE, True)
file = split_audiofile(file, 8, SAMPLE_RATE)
file = audio_splits_to_spectograms(file, 2048)
file = dimesion_for_VAE(file)
save_training_data(file, DATA_PATH+"/training_v1.npy")