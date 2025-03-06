# Torch
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
# Utils
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import librosa, os, logging, time, soundfile

#Logging
LIGHT_DEBUG: int = 15

def light_debug(self, message, *args, **kws) -> None:
    if self.isEnabledFor(LIGHT_DEBUG):
        self._log(LIGHT_DEBUG, message, args, **kws)


logging.addLevelName(LIGHT_DEBUG, "LIGHT_DEBUG")
logging.Logger.light_debug = light_debug

logger = logging.getLogger(__name__)

# Loading and Saving Data
def load_audio_file(path: str, sample_rate: int = 44100, to_mono: bool = True) -> ndarray:
    audio, current_sample_rate = librosa.load(path, sr=None, mono=to_mono) 
    if current_sample_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=sample_rate)
    logger.light_debug(f"Loaded audio form {path} of dimensions: {audio.shape}, sr: {sample_rate}")
    return audio

def load_spectogram(path: str) -> ndarray:
    spectogram: ndarray = np.load(path)["stft"]
    logger.light_debug(f"Spectogram loaded from {path} of shape: {spectogram.shape}")
    return spectogram

def save_spectogram(spectogram: ndarray, path: str) -> ndarray:
    np.savez_compressed(path, stft=spectogram)
    logger.light_debug(f"Saved spectogram to:{path}")

def save_training_data(data: ndarray, path: str) -> None:
    np.save(path, data)
    logger.light_debug(f"Saved ndarray to:{path}")

def load_training_data(path: str) -> ndarray:
    data: ndarray= np.load(path)
    logger.light_debug(f"Ndarray loaded from {path} of shape: {data.shape}")
    return data

def save_audio_file(audio: ndarray, path: str, sample_rate: int = 44100) -> None:
    if audio.dtype != np.int16:
        audio = normalize(audio, -0.99999, 0.99999)
    soundfile.write(path, audio, sample_rate)
    logger.light_debug(f"Saved file to:{path}")

def get_filenames_from_folder(path: str, filetype: str = None) -> list:
    if filetype != None:
        files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(filetype)]
    else:
        files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    logger.light_debug(f"Got filenames {files} from {path}")
    return files

def path_to_remote_path(path: str, is_remote: bool = True) -> str:
    if is_remote:
        return path[3:]
    else: return path
# Other Manipulations
def split_audiofile(audio: ndarray, time: int, sample_rate: int = 44100, overlap_s: int = 0) -> ndarray:
    samples: int = sample_rate * time
    samples_overlap: int = sample_rate * overlap_s
    if overlap_s == 0:
        pad: int = len(audio) % samples
        audio = np.pad(audio, (0, samples - pad))
        data = np.array(np.split(audio, len(audio) // samples))
    else:
        data: list = []
        for i in range(0, audio.shape[0] - samples + 1, samples - samples_overlap):
            split: ndarray = audio[i: i + samples]
            if split.shape[0] != samples:
                split = np.pad(split, (0, samples - split.shape[0]))
            data.append(split)
        data = np.array(data)

    logger.light_debug(f"Split audio to: {data.shape}")
    return data

def create_dataloader(data: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

# Spectograms
def audio_to_spectrogram(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    logger.light_debug("Started STFT")
    stft = librosa.stft(audio, n_fft=len_fft, hop_length=hop_length)
    spec = np.abs(stft)
    if log:
        spec = librosa.amplitude_to_db(spec)
    logger.light_debug(f"Created spectogram: {spec.shape}")
    return spec

def audio_splits_to_spectograms(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    logger.light_debug("Started STFT on splits")
    specs: list = []
    for i,split in enumerate(audio):
        stft = librosa.stft(split, n_fft=len_fft, hop_length=hop_length)
        spec = np.abs(stft)
        if log:
            spec = librosa.amplitude_to_db(spec)
        specs.append(spec)
        if (i + 1) % 10 == 0 and logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Processed Splits: {i + 1}", end='')
    if logger.getEffectiveLevel() == LIGHT_DEBUG:
        print()
    specs: ndarray = np.array(specs)
    logger.light_debug(f"Created spectograms of splits: {specs.shape}")
    return specs

def spectrogram_to_audio(spec: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    logger.light_debug("Started GL")
    if spec.shape[0] != len_fft // 2 + 1:
        spec = np.pad(spec, ((0, abs((len_fft // 2 + 1) - spec.shape[0])), (0, 0)), mode='constant')
    if log:
        spec = librosa.db_to_amplitude(spec)
    audio: ndarray = librosa.griffinlim(spec, n_fft=len_fft, hop_length=hop_length)
    audio = normalize(audio, -0.99999, 0.99999)
    logger.light_debug(f"Reconstructed audio: {audio.shape}")
    return audio

def normalize(data: ndarray, min_val: float = 0, max_val: float = 1) -> ndarray:
    min_data: float = np.min(data)
    max_data: float = np.max(data)
    scaled_data: ndarray = (data - min_data) / (max_data - min_data)
    normalized_data: ndarray = scaled_data * (max_val - min_val) + min_val
    logger.light_debug(f"Normalized to range: [{min_val},{max_val}]")
    return normalized_data

def unnormalize(data: ndarray, min_val: float = -50, max_val: float = 50) -> ndarray:
    min_data: float = np.min(data)
    max_data: float = np.max(data)
    scaled_data: ndarray = (data - min_data) / (max_data - min_data)
    normalized_data: ndarray = scaled_data * (max_val - min_val) + min_val
    logger.light_debug(f"Unnormalized to range: [{min_val},{max_val}]")
    return normalized_data

def dimension_for_VAE(data: ndarray) -> ndarray:
    if data.shape[-1] % 32 != 0:
        data = data[...,:(data.shape[-1] // 32) * 32]
    if data.shape[-2] % 32 != 0:
        data = data[...,:(data.shape[-2] // 32) * 32, :]
    return data

def audio_to_mel_spectogram(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, sample_rate: int = 44100, log: bool = True, min_freq: int = 30, n_mels: int = 128) -> ndarray:
    logger.light_debug("Started Mel-Spec")
    spec = librosa.feature.melspectrogram(y=audio, n_fft=len_fft, hop_length=hop_length, sr=sample_rate, fmin=min_freq, n_mels=n_mels)
    if log:
        spec = librosa.amplitude_to_db(spec)
    logger.light_debug(f"Created mel-spectogram: {spec.shape}")
    return spec

def audio_splits_to_mel_spectograms(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, sample_rate: int = 44100, log: bool = True, min_freq: int = 30, n_mels: int = 128) -> ndarray:
    logger.light_debug("Started Mel-Spec on splits")
    specs: list = []
    for i,split in enumerate(audio):
        spec = librosa.feature.melspectrogram(y=split, n_fft=len_fft, hop_length=hop_length, sr=sample_rate, fmin=min_freq, n_mels=n_mels)
        if log:
            spec = librosa.amplitude_to_db(spec)
        specs.append(spec)
        if (i + 1) % 10 == 0 and logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Processed Splits: {i + 1}", end='')
    if logger.getEffectiveLevel() == LIGHT_DEBUG:
        print()
    specs: ndarray = np.array(specs)
    logger.light_debug(f"Created me-spectograms of splits: {specs.shape}")
    return specs

def mel_spectrogram_to_audio(spec: ndarray, len_fft: int = 4096, hop_length: int = 512, sample_rate: int = 44100, log: bool = True) -> ndarray:
    logger.light_debug("Started GL")
    if spec.shape[0] != len_fft // 2 + 1:
        spec = np.pad(spec, ((0, abs((len_fft // 2 + 1) - spec.shape[0])), (0, 0)), mode='constant')
    if log:
        spec = librosa.db_to_amplitude(spec)
    audio: ndarray = librosa.feature.inverse.mel_to_audio(spec, sr=sample_rate, n_fft=len_fft, hop_length=hop_length)
    audio = normalize(audio, -0.99999, 0.99999)
    logger.light_debug(f"Reconstructed audio: {audio.shape}")
    return audio


# Visualize Data
def scatter_plot(data_x: ndarray, data_y: ndarray = None, x_label: str = "Epoch", y_label: str = "Lr", color: str = "blue", switch_x_y: bool = True) -> None:
    if data_y is None:
        data_y = np.arange(len(data_x))
    if switch_x_y:
        plt.scatter(data_y, data_x, c=color)
        plt.xlabel = y_label
        plt.ylabel = x_label
    else:
        plt.scatter(data_x, data_y, c=color)
        plt.xlabel = x_label
        plt.ylabel = y_label
    plt.show()

def visualize_spectogram(spectogram: ndarray, sample_rate: int = 44100, len_fft: int = 4096) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectogram, sr=sample_rate, n_fft=len_fft)
    plt.show()
# Torch utils
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Audio_Data(Dataset):
    def __init__(self, data: ndarray, labels: ndarray = None, dtype: torch.dtype = torch.float32) -> None:
        self.data = torch.tensor(data, dtype=dtype)
        if labels != None:
            self.labels = torch.tensor(labels, dtype=dtype) 
        else:
            self.labels = torch.tensor(data, dtype=dtype)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

