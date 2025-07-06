# Torch
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import optuna
# Utils
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import librosa, os, logging, time, soundfile, midi2audio, tempfile
from midi2audio import FluidSynth
from scipy.signal import butter, lfilter
from typing import Callable
#Logging
LIGHT_DEBUG: int = 15

def light_debug(self, message, *args, **kws) -> None:
    if self.isEnabledFor(LIGHT_DEBUG):
        self._log(LIGHT_DEBUG, message, args, **kws)


logging.addLevelName(LIGHT_DEBUG, "LIGHT_DEBUG")
logging.Logger.light_debug = light_debug

logger = logging.getLogger(__name__)

# Loading and Saving Data
def load_audio_file(path: str, sr: int = 44100, to_mono: bool = True) -> ndarray:
    """Loads an audio file, casts it to the given sample rate and returns a numpy array.

    Args:
        path (str): The file path.
        sr (int, optional): The sample rate to cast the audio to. Defaults to 44100.
        to_mono (bool, optional): If true converts audio to mono if stereo. Defaults to True.

    Returns:
        ndarray: returns a 1-D if mono or 2-D if stereo ndarray
    """
    audio, current_sr = librosa.load(path, sr=None, mono=to_mono) 
    if current_sr != sr:
        audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)
    logger.light_debug(f"Loaded audio form {path} of dimensions: {audio.shape}, sr: {sr}")
    return audio

def load_spectrogram(path: str) -> ndarray:
    """Loads a spectrogram to np array.

    Args:
        path (str): Path to spectrogram.

    Returns:
        ndarray: _description_
    """
    spectrogram: ndarray = np.load(path)["stft"]
    logger.light_debug(f"spectrogram loaded from {path} of shape: {spectrogram.shape}")
    return spectrogram

def save_spectrogram(spectrogram: ndarray, path: str) -> None:
    """Saves spectrogram to path.

    Args:
        spectrogram (ndarray): A spectrogram.
        path (str): Path to save to.
    """
    np.savez_compressed(path, stft=spectrogram)
    logger.light_debug(f"Saved spectrogram to:{path}")

def save_training_data(data: ndarray, path: str) -> None:
    """Saves numpy array to path.

    Args:
        data (ndarray): Numpy array.
        path (str): Filepath.
    """
    if not path.endswith(".npy"):
        path += ".npy"
    np.save(path, data)
    logger.light_debug(f"Saved ndarray to:{path}")

def load_training_data(path: str) -> ndarray:
    """Loads an numpy array from path.

    Args:
        path (str): Filepath.

    Returns:
        ndarray: The loaded data.
    """
    if not path.endswith(".npy"):
        path += ".npy"
    data: ndarray= np.load(path)
    logger.light_debug(f"Ndarray loaded from {path} of shape: {data.shape}")
    return data

def save_audio_file(audio: ndarray, path: str, sr: int = 44100) -> None:
    """Saves an numpy array as audiofile, normalizes the audio if needed.
    Args:
        audio (ndarray): Audio data, 1-D or 2-D array.
        path (str): Filepath has to end with the filetype e.g. .wav.
        sr (int, optional): Samplerate of the audio. Defaults to 44100.
    """
    if audio.dtype != np.int16:
        audio = normalize(audio, -0.99999, 0.99999)
    soundfile.write(path, audio, sr)
    logger.light_debug(f"Saved file to:{path}")

def get_filenames_from_folder(path: str, filetype: str = None) -> list:
    """Fetches all filenames froma given folder, if filetype is speciefied only filenames of that type are returned.

    Args:
        path (str): Folderpath.
        filetype (str, optional): If passed, filters folder for this filetype e.g. .wav. Defaults to None.

    Returns:
        list: Returns a list of found filenames.
    """
    if filetype != None:
        files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(filetype)]
    else:
        files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    logger.light_debug(f"Got filenames {files} from {path}")
    return files

def path_to_remote_path(path: str, is_remote: bool = True) -> str:
    """If is_remote = True removes the ../ from the path to fit the remote kernel structure.

    Args:
        path (str): File or folderpath.
        is_remote (bool, optional): True if working on a remote kernel. Defaults to True.

    Returns:
        str: Modified filepath.
    """
    if is_remote:
        return path[3:]
    else: return path

def del_if_exists(path: str) -> None:
    """Deletes a file or directory if it exists, given its path.

    Args:
        path (str): filepath.
    """
    if os.path.exists(path):
        os.remove(path)
        logger.light_debug(f"{path} deleted")
    else:
        logger.light_debug(f"{path} could not be deleted")

# Other Manipulations
def split_audiofile(audio: ndarray, time: float, sr: int = 44100, overlap_s: float = 0) -> ndarray:
    """Splits audio into samples of length time, with an optional overlap. Pads the last file with zeroes if necessary.

    Args:
        audio (ndarray): Audiofile.
        time (float): Sample length in s.
        sr (int, optional): Sample rate. Defaults to 44100.
        overlap_s (float, optional): Overlap of the samples in s. Defaults to 0.

    Returns:
        ndarray: Nd-array containing the audiosplits.
    """
    samples: int = int(sr * time)
    samples_overlap: int = int(sr * overlap_s)
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

def midi2ndarray(path: str, sr: int = 32000, sf_path: str = "UprightPiano.sf2") -> ndarray:
    fs = FluidSynth(sf_path, sample_rate=sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        fs.midi_to_audio(path, temp_wav.name)
        print("Converted")
        audio, _ = soundfile.read(temp_wav.name)
        os.unlink(temp_wav.name)
    return audio

def patch_safe_audio(audio_lst: ndarray, path: str, sr: int = 32000) -> None:
    audio = np.concatenate(audio_lst)
    save_audio_file(audio, path, sr)

def create_dataloader(data: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 1) -> DataLoader:
    """Creates a torch dataloader. Optionally shuffles the data.

    Args:
        data (Dataset): The dataloader data.
        batch_size (int): Batch size.
        shuffle (bool, optional): If true schuffles data. Defaults to False.
        num_workers(int, optional): Sets number of workers. Defaults to 1.

    Returns:
        DataLoader: The torch Dataloader.
    """
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# spectrograms
def audio_to_spectrogram(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    """Converts audio to stft spectrograms and optionally converts the spectrograms to log scale.

    Args:
        audio (ndarray): Audio data._
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        log (bool, optional): If true converts spectrogram to log scale. Defaults to True.

    Returns:
        ndarray: An stft spectrogram.
    """
    logger.light_debug("Started STFT")
    stft = librosa.stft(audio, n_fft=len_fft, hop_length=hop_length)
    spec = np.abs(stft)
    if log:
        spec = np.log(spec + 1e-6)
    logger.light_debug(f"Created spectrogram: {spec.shape}")
    return spec

def audio_splits_to_spectrograms(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    """Converts a numpy array containing multiple samples to stft spectrograms and optionally converts the spectrograms to log scale.

    Args:
        audio (ndarray): Audio data._
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        log (bool, optional): If true converts spectrogram to log scale. Defaults to True.

    Returns:
        ndarray: An numpy array containing the stft spectrograms.
    """
    logger.light_debug("Started STFT on splits")
    specs: list = []
    for i,split in enumerate(audio):
        stft = librosa.stft(split, n_fft=len_fft, hop_length=hop_length)
        spec = np.abs(stft)
        if log:
            spec = np.log(spec + 1e-6)
        specs.append(spec)
        if (i + 1) % 10 == 0 and logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Processed Splits: {i + 1}", end='')
    if logger.getEffectiveLevel() == LIGHT_DEBUG:
        print()
    specs: ndarray = np.array(specs)
    logger.light_debug(f"Created spectrograms of splits: {specs.shape}")
    return specs

def spectrogram_to_audio(spec: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    """Uses Griffinlim to convert a spectrogram to audio.

    Args:
        spec (ndarray): An STFT spectrogram.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        log (bool, optional): Set to true if spectrogram is in log scale. Defaults to True.

    Returns:
        ndarray: An audio file
    """
    logger.light_debug("Started GL")
    if spec.shape[0] != len_fft // 2 + 1:
        spec = np.pad(spec, ((0, abs((len_fft // 2 + 1) - spec.shape[0])), (0, 0)), mode='constant')
    if log:
        spec = np.exp(spec)
    audio: ndarray = librosa.griffinlim(spec, n_fft=len_fft, hop_length=hop_length)
    audio = normalize(audio, -0.99999, 0.99999)
    logger.light_debug(f"Reconstructed audio: {audio.shape}")
    return audio

def normalize(data: ndarray, min_val: float = -1, max_val: float = 1) -> ndarray:
    """Maps data to a given range.

    Args:
        data (ndarray): Data.
        min_val (float, optional): Min value. Defaults to 0.
        max_val (float, optional): Max value. Defaults to 1.

    Returns:
        ndarray: Mapped data.
    """
    min_data: float = np.min(data)
    max_data: float = np.max(data)
    scaled_data: ndarray = (data - min_data) / (max_data - min_data)
    normalized_data: ndarray = scaled_data * (max_val - min_val) + min_val
    logger.light_debug(f"Normalized to range: [{min_val},{max_val}]")
    return normalized_data

def normalize_filewise(data: ndarray, min_val: float = -1, max_val: float = 1) -> ndarray:
    """Normalizes an array containg the data as sub arays, sub array wise. E.g. when taken from audiosplits it normalizes each split to the range given. 

    Args:
        data (ndarray): Data
        min_val (float, optional): Min value. Defaults to -1.
        max_val (float, optional): Max value. Defaults to 1.

    Returns:
        ndarray: Filenormalized array.
    """
    normalized_data: ndarray = np.zeros_like(data)
    for i, file in enumerate(data):
        min_file: float = np.min(file)
        max_file: float = np.max(file)
        scaled_file: ndarray = (file - min_file) / (max_file - min_file)
        normalized_file: ndarray = scaled_file * (max_val - min_val) + min_val
        normalized_data[i] = normalized_file
    logger.light_debug(f"Normalized to range: [{min_val},{max_val}]")
    return normalized_data

def unnormalize(data: ndarray, min_val: float = -50, max_val: float = 50) -> ndarray:
    """Convenience function, does the same as normalize but is set to unnormalize to spectrogram range.

    Args:
        data (ndarray): Data.
        min_val (float, optional): Min value. Defaults to -50.
        max_val (float, optional): Max value. Defaults to 50.

    Returns:
        ndarray: Mapped data.
    """
    min_data: float = np.min(data)
    max_data: float = np.max(data)
    scaled_data: ndarray = (data - min_data) / (max_data - min_data)
    normalized_data: ndarray = scaled_data * (max_val - min_val) + min_val
    logger.light_debug(f"Unnormalized to range: [{min_val},{max_val}]")
    return normalized_data

def dimension_for_VAE(data: ndarray) -> ndarray:
    """Dimensions data for easier use. Crops data to be of a size divisible by 32.

    Args:
        data (ndarray): Data.

    Returns:
        ndarray: Cropped data. 
    """
    if data.shape[-1] % 32 != 0:
        data = data[...,:(data.shape[-1] // 32) * 32]
    if data.shape[-2] % 32 != 0:
        data = data[...,:(data.shape[-2] // 32) * 32, :]
    return data

def audio_to_mel_spectrogram(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, sr: int = 44100, log: bool = True, min_freq: int = 30, n_mels: int = 128) -> ndarray:
    """Creates a mel spectrogram from an audio file.

    Args:
        audio (ndarray): Audio data.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sr (int, optional): Sample rate. Defaults to 44100.
        log (bool, optional): If true converts spectrogram to log scale. Defaults to True.
        min_freq (int, optional): Sets the minimal frequency represented by spectrogram. Defaults to 30.
        n_mels (int, optional): Number of mel bins. Defaults to 128.

    Returns:
        ndarray: Mel spectrogram.
    """
    logger.light_debug("Started Mel-Spec")
    spec = librosa.feature.melspectrogram(y=audio, n_fft=len_fft, hop_length=hop_length, sr=sr, fmin=min_freq, n_mels=n_mels)
    if log:
        spec = np.log(spec + 1e-6)
    logger.light_debug(f"Created mel-spectrogram: {spec.shape}")
    return spec

def audio_splits_to_mel_spectrograms(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, sr: int = 44100, log: bool = True, min_freq: int = 30, n_mels: int = 128) -> ndarray:
    """Creates mel spectrograms from audio samples.

    Args:
        audio (ndarray): Audio samples.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sr (int, optional): Sample rate. Defaults to 44100.
        log (bool, optional): If true converts spectrogram to log scale. Defaults to True.
        min_freq (int, optional): Sets the minimal frequency represented by spectrogram. Defaults to 30.
        n_mels (int, optional): Number of mel bins. Defaults to 128.

    Returns:
        ndarray: Mel spectrograms.
    """
    logger.light_debug("Started Mel-Spec on splits")
    specs: list = []
    for i,split in enumerate(audio):
        spec = librosa.feature.melspectrogram(y=split, n_fft=len_fft, hop_length=hop_length, sr=sr, fmin=min_freq, n_mels=n_mels)
        if log:
            spec = np.log(spec + 1e-6)
        specs.append(spec)
        if (i + 1) % 10 == 0 and logger.getEffectiveLevel() == LIGHT_DEBUG:
            print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Processed Splits: {i + 1}", end='')
    if logger.getEffectiveLevel() == LIGHT_DEBUG:
        print()
    specs: ndarray = np.array(specs)
    logger.light_debug(f"Created Mel-spectrograms of splits: {specs.shape}")
    return specs

def mel_spectrogram_to_audio(spec: ndarray, len_fft: int = 4096, hop_length: int = 512, sr: int = 44100, log: bool = True) -> ndarray:
    """Uses Griffinlim to convert a mel-spectrogram to audio.

    Args:
        spec (ndarray): An STFT spectrogram.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sr (int, optional): Sample rate. Defaults to 44100.
        log (bool, optional): Set to true if spectrogram is in log scale. Defaults to True.

    Returns:
        ndarray: An audio file
    """
    logger.light_debug("Started GL")
    if log:
        spec = np.exp(spec)
    audio: ndarray = librosa.feature.inverse.mel_to_audio(spec, sr=sr, n_fft=len_fft, hop_length=hop_length)
    audio = normalize(audio, -0.99999, 0.99999)
    logger.light_debug(f"Reconstructed audio: {audio.shape}")
    return audio

def sdr(audio: ndarray, sr: int = 44100, cutoff: int = 4000) -> float:
    """
    Approximate SDR without a reference by comparing audio to a low-pass filtered version.
    
    Args:
        audio (np.ndarray): Generated waveform.
        sr (int): Sampling rate in Hz.
        cutoff (float): Cutoff frequency for low-pass filter (Hz).
    
    Returns:
        float: SDR in dB.
    """
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    smoothed_audio = lfilter(b, a, audio)
    
    signal_power = np.mean(np.square(smoothed_audio))

    distortion = audio - smoothed_audio
    distortion_power = np.mean(np.square(distortion))
    
    if distortion_power == 0:
        return float('inf')
    
    sdr = 10 * np.log10(signal_power / distortion_power)
    logger.light_debug("Calculated SDR")
    return sdr

def spectral_convergence(spectrogram: ndarray, len_fft: int = 4096, hop_length: int = 512, sr: int = 44100, log: bool = True) -> float:
    """Measure spectral convergence of a generated spectrogram.

    Args:
        spectrogram (ndarray): spectrogram.
        len_fft (int, optional): STFT FFT length. Defaults to 1024.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sr (int, optional): Samplerate. Defaults to 44100.
        log (bool, optional): Set to true if spectrogram is in log scale. Defaults to True.

    Returns:
        float: Spectral convergence score.
    """
    audio = spectrogram_to_audio(spectrogram, len_fft, hop_length, log)
    
    # Recompute STFT from the waveform
    reconstructed_spec = np.abs(librosa.stft(audio, n_fft=len_fft, hop_length=hop_length))
    
    # Pad or trim to match original shape (224x416)
    reconstructed_spec = reconstructed_spec[:spectrogram.shape[-2], :spectrogram.shape[-1]]
    
    diff = np.mean((spectrogram - reconstructed_spec) ** 2)
    norm = np.mean(spectrogram ** 2)

    if norm == 0:
        return float('inf')
    
    return np.sqrt(diff / norm)

def flatten(data: ndarray) -> ndarray:
    """Flattens a 3D (N,H,W) to 2d (N,H*W).

    Args:
        data (ndarray): Array to flatten.

    Returns:
        ndarray: Flattened array.
    """
    N, H, W = data.shape
    output: ndarray = np.zeros((N, H * W))
    for i, file in enumerate(data):
        output[i] = file.flatten()
    return output


# Visualize Data
def scatter_plot(data_x: ndarray, data_y: ndarray = None, x_label: str = "Epoch", y_label: str = "Lr", color: str = "blue", switch_x_y: bool = True) -> None:
    """Visualizes data as a scatterplot.

    Args:
        data_x (ndarray): X data.
        data_y (ndarray, optional): Y data. If not given numerates X. Defaults to None.
        x_label (str, optional): X-Axis label. Defaults to "Epoch".
        y_label (str, optional): Y-Axis label. Defaults to "Lr".
        color (str, optional): Color of the plot. Defaults to "blue".
        switch_x_y (bool, optional): If true switches x and y axis. Defaults to True.
    """
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

def visualize_spectrogram(spectrogram: ndarray, sr: int = 44100, len_fft: int = 4096) -> None:
    """Plots a spectrogram.

    Args:
        spectrogram (ndarray): spectrogram.
        sr (int, optional): Sample rate. Defaults to 44100.
        len_fft (int, optional): STFT FTT length. Defaults to 4096.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, n_fft=len_fft)
    plt.show()


# Torch utils
def count_parameters(model: nn.Module) -> str:
    """Counts all parameters of NN module. 

    Args:
        model (nn.Module): A torch nn.Module.
    Returns:
        int: Number of parameters.
    """
    suffixes: dict = {1e9:"B", 1e6:"M", 1e3:"k", 1e0:""}
    n =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    for key, val in suffixes.items():
        if n / key > 1:
            n = round(n / key, 3)
            return f"~{str(n)[:5]}{val}"
class Audio_Data(Dataset):
    def __init__(self, data: ndarray, labels: ndarray = None, dt: torch.dtype = torch.float32) -> None:
        """Creates a torch dataloader. 

        Args:
            data (ndarray): Data.
            labels (ndarray, optional): If labels are not given labels are set to be the data. Defaults to None.
            dtype (torch.dtype, optional): Datatype. Defaults to torch.float32.
        """
        if type(data) is not  Tensor:
            data: Tensor = torch.tensor(data)
        if type(labels) is not Tensor and labels is not None:
            labels: Tensor = torch.tensor(labels)
        
        if labels is not None:
            self.labels = labels.to(dtype=dt) 
        else:
            self.labels = data.to(dtype=dt) 
        self.data = data.to(dtype=dt)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class Trainer():
    def __init__(self, model: nn.Module, optimizer: Optimizer = None, lr_scheduler: _LRScheduler = None, device: str = "cpu", embed_fun: Callable | None = None, embed_dim: int | None = None, n_dims: int = 1)-> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.embed_fun = embed_fun
        self.embed_dim = embed_dim
        self.n_dims = n_dims

    def train(self, train_dataset: DataLoader, n_epochs: int, full_model_path: str, checkpoint_freq: int = 0, val_dataset: DataLoader = None, patience: int = -1, gradient_clip_norm: float| None = None, gradient_clip_val: float | None = None, sample_freq: int | None = None) -> tuple[list[float], list[float] | None]:
        logger.info(f"Training started on {self.device}")
        if self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler() #This is obsolete and the other version would work for cuda aswell, but paperspace does not support the other version yet
        else:
            self.scaler = torch.amp.GradScaler(device=self.device)
        loss_list: list = []
        val_loss_list: list = []
        total_time: float = 0.0
        best_loss = float('inf')
        epochs_no_improve: int = 0

        self.model.train()
        for e in range(0, n_epochs):
            total_loss: float = 0
            validation_loss: float = 0
            start_time: float = time.time()

            for b_idx, (x, y) in enumerate(train_dataset):
                self.optimizer.zero_grad()
                if x.dim() == self.n_dims + 1:
                    x = x.to(self.device).unsqueeze(1)
                else:
                    x = x.to(self.device)
                with torch.autocast(device_type=self.device):
                    loss = self.model(x)

                loss.backward()

                total_loss += loss.item()
                if np.isnan(loss.item()):
                    logger.info("Breaking due to NaN loss.")
                    break

                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
                if gradient_clip_val is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), gradient_clip_val)
                
                self.optimizer.step()

                if logger.getEffectiveLevel() == LIGHT_DEBUG:
                    current_batch = b_idx + 1
                    all_params = torch.cat([param.view(-1) for param in self.model.parameters()])
                    print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT_DEBUG - Batch {current_batch:03d}/{len(train_dataset):03d} Loss: {loss.item():.3f} Min/Max params: {torch.min(all_params):.3f}, {torch.max(all_params):.3f}", end='', flush=True)
            else:
                if logger.getEffectiveLevel() == LIGHT_DEBUG:
                    print(flush=True)

                if val_dataset is not None:
                    self.model.eval()
                    for (x, y) in val_dataset:
                        if x.dim() == self.n_dims + 1:
                            x = x.to(self.device).unsqueeze(1)
                        else:
                            x = x.to(self.device)
                        with torch.no_grad():
                            loss = self.model(x)
                            validation_loss += loss.item()
                    validation_loss = validation_loss / len(val_dataset)
                    val_loss_list.append(validation_loss)
                    self.model.train()

                avg_loss = total_loss / len(train_dataset)
                loss_list.append(avg_loss)

                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, (optim.lr_scheduler.ReduceLROnPlateau)):
                        self.lr_scheduler.step(avg_loss)
                    else:
                        self.lr_scheduler.step()

                if patience > 0:
                    if avg_loss < best_loss:
                        epochs_no_improve = 0
                        best_loss = avg_loss
                    else:
                        epochs_no_improve += 1
                
                if epochs_no_improve >= patience and patience != -1:
                    logger.info(f"Early stopping at epoch {e + 1}: Loss has not improved for {patience} epochs")
                    break
                
                if sample_freq is not None and (e + 1) % sample_freq == 0:
                    x, _ = next(iter(train_dataset))
                    if x.dim() == self.n_dims + 1:
                        x = x.unsqueeze(1)
                    c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
                    if self.n_dims == 2:
                        self.sample(2, [2, c, h, w], 10, True)
                    else:
                        self.sample(2,[2, h, w], 10, True)
                epoch_time = time.time() - start_time
                total_time += epoch_time
                remaining_time = int((total_time / (e + 1)) * (n_epochs - e - 1))
                val_loss_str = f" Avg. val. Loss: {validation_loss:.5e}" if val_dataset is not None else ""

                logger.info(f"Epoch {e + 1:03d}: Avg. Loss: {avg_loss:.5e}{val_loss_str} Remaining Time: {remaining_time // 3600:02d}h {(remaining_time % 3600) // 60:02d}min {round(remaining_time % 60):02d}s LR: {self.optimizer.param_groups[0]['lr']:.5e} ")
                
                if checkpoint_freq > 0 and (e + 1) % checkpoint_freq == 0:
                    checkpoint_path: str = f"{full_model_path[:-4]}_epoch_{e + 1:03d}.pth"
                    torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict(), "scheduler": self.lr_scheduler.state_dict(), "epoch": e + 1}, checkpoint_path)
                    if e + 1 != checkpoint_freq:
                        last_path: str = f"{full_model_path[:-4]}_epoch_{(e + 1) - checkpoint_freq:03d}.pth"
                        del_if_exists(last_path)
                    logger.light_debug(f"Checkpoint saved model to {checkpoint_path}")
                continue
            break


        torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict(), "scheduler": self.lr_scheduler.state_dict(), "epoch": e + 1}, full_model_path)

        logger.light_debug(f"Saved model to {full_model_path}")

        if checkpoint_freq > 0:
            checkpoint_path: str = f"{full_model_path[:-4]}_epoch_{e + 1 - ((e + 1) % checkpoint_freq):03d}.pth"
            del_if_exists(checkpoint_path)
        
        return loss_list, val_loss_list
    
    def _convert_to_item_list(self, x: Tensor) -> ndarray:
        if x.ndim == 4:
            x = torch.squeeze(x, 1)
        return x.cpu().numpy()
    
    def save_architecture(self, tensor_dim: list, path: str) -> None:
        self.model.eval()
        with torch.no_grad():
            example_x = torch.randn(*tensor_dim).to(self.device)
            script_model = torch.jit.trace(self.model,example_x, check_trace=False)
        torch.jit.save(script_model, path)
        self.model.train()
    
    def sample(self, n_samples: int, tensor_dim: list, n_steps: int = 20, visualize: bool = False) -> ndarray:
        if self.n_dims == 2:
            noise = torch.randn(n_samples, tensor_dim[-3], tensor_dim[-2], tensor_dim[-1]).to(self.device)
        else:
            noise = torch.randn(n_samples, tensor_dim[-2], tensor_dim[-1]).to(self.device)
        self.model.eval()
        samples = self._convert_to_item_list(self.model.sample(noise, num_steps=n_steps))
        samples = normalize(samples, -1, 1)
        self.model.train()
        if visualize:
            self.visualize_samples(samples)
        return samples
    
    def sample_voc(self, spec: ndarray, n_steps: int = 20) -> ndarray:
        if spec.ndim == 2:
            spec = np.reshape(spec, [1, 1, spec.shape[-2], spec.shape[-1]])
        if spec.ndim == 3:
            spec = np.reshape(spec, [spec.shape[0], 1, spec.shape[-2], spec.shape[-1]])
        spec = torch.tensor(spec)
        wave = self.model.sample(spec, num_steps=n_steps)
        return wave.cpu().numpy()
    
    def sample_AE(self, seed: ndarray, n_steps: int = 20) -> ndarray:
        if seed.ndim == 2:
            seed = np.reshape(seed, [seed.shape[0], 1, seed.shape[1]])
        elif seed.ndim == 1:
            seed = np.reshape(seed, [1, 1, seed.shape[0]])
        seed = torch.tensor(seed).to(self.device)
        latent = self.model.encode(seed)
        sample = self.model.decode(latent, num_steps=n_steps).cpu().numpy()
        return sample

    def visualize_samples(self, samples: ndarray) -> None:
        for sample in samples:
            visualize_spectrogram(normalize(sample, -1, 1), sr=32000)
    
    def save_samples(self, samples: ndarray, file_path_name: str, sr: int = 32000, len_fft: int = 480, len_hop: int = 288) -> None:
        for i, sample in enumerate(samples):
            audio = spectrogram_to_audio(unnormalize(sample), len_fft, len_hop)
            save_audio_file(audio, f"{file_path_name}_{i:02d}.wav", sr=sr)
    
    def get_audio_metrics(self, samples: ndarray, original_dataset: ndarray = None, sr: int = 32000, len_fft: int = 480, len_hop: int = 288) -> None:
        avg_sample_spectral_conv: float = 0
        avg_sample_spectral_cent: float = 0
        avg_true_spectral_conv: float = 0
        avg_true_spectral_cent: float = 0
        for sample in samples:
            avg_sample_spectral_conv += spectral_convergence(sample, sr=sr, len_fft=len_fft, hop_length=len_hop)
            avg_sample_spectral_cent += np.mean(librosa.feature.spectral_centroid(y=sample, sr=sr))

        if original_dataset is not None:
            indices: list = np.random.choice(len(original_dataset), 100)
            for idx in indices:
                sample = original_dataset[idx]
                avg_true_spectral_conv += spectral_convergence(sample, sr=sr, len_fft=len_fft, hop_length=len_hop)
                avg_true_spectral_cent += np.mean(librosa.feature.spectral_centroid(y=sample, sr=sr))

        n = len(samples)
        avg_sample_spectral_conv = avg_sample_spectral_conv / n
        avg_sample_spectral_cent = avg_sample_spectral_cent / n
        avg_true_spectral_conv = avg_true_spectral_conv / 100
        avg_true_spectral_cent = avg_true_spectral_cent / 100
        metrics_str: str = f"Spectral Convergence Samples/Real: {avg_sample_spectral_conv:.3f}, {avg_true_spectral_conv:.3f} Spectral Centroid Samples/Real: {avg_sample_spectral_cent:.3f} Hz, {avg_true_spectral_cent:.3f} Hz" if original_dataset is not None else f"Spectral Convergence Samples: {avg_sample_spectral_conv:.3f} Spectral Centroid Samples: {avg_sample_spectral_cent:.3f} Hz"
        print(metrics_str)
