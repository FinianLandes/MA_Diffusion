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
from scipy.signal import butter, lfilter
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
    """Loads an audio file, casts it to the given sample rate and returns a numpy array.

    Args:
        path (str): The file path.
        sample_rate (int, optional): The sample rate to cast the audio to. Defaults to 44100.
        to_mono (bool, optional): If true converts audio to mono if stereo. Defaults to True.

    Returns:
        ndarray: returns a 1-D if mono or 2-D if stereo ndarray
    """
    audio, current_sample_rate = librosa.load(path, sr=None, mono=to_mono) 
    if current_sample_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=sample_rate)
    logger.light_debug(f"Loaded audio form {path} of dimensions: {audio.shape}, sr: {sample_rate}")
    return audio

def load_spectogram(path: str) -> ndarray:
    """Loads a spectogram to np array.

    Args:
        path (str): Path to spectogram.

    Returns:
        ndarray: _description_
    """
    spectogram: ndarray = np.load(path)["stft"]
    logger.light_debug(f"Spectogram loaded from {path} of shape: {spectogram.shape}")
    return spectogram

def save_spectogram(spectogram: ndarray, path: str) -> None:
    """Saves Spectogram to path.

    Args:
        spectogram (ndarray): A Spectogram.
        path (str): Path to save to.
    """
    np.savez_compressed(path, stft=spectogram)
    logger.light_debug(f"Saved spectogram to:{path}")

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

def save_audio_file(audio: ndarray, path: str, sample_rate: int = 44100) -> None:
    """Saves an numpy array as audiofile, normalizes the audio if needed.
    Args:
        audio (ndarray): Audio data, 1-D or 2-D array.
        path (str): Filepath has to end with the filetype e.g. .wav.
        sample_rate (int, optional): Samplerate of the audio. Defaults to 44100.
    """
    if audio.dtype != np.int16:
        audio = normalize(audio, -0.99999, 0.99999)
    soundfile.write(path, audio, sample_rate)
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
def split_audiofile(audio: ndarray, time: int, sample_rate: int = 44100, overlap_s: int = 0) -> ndarray:
    """Splits audio into samples of length time, with an optional overlap. Pads the last file with zeroes if necessary.

    Args:
        audio (ndarray): Audiofile.
        time (int): Sample length in s.
        sample_rate (int, optional): Sample rate. Defaults to 44100.
        overlap_s (int, optional): Overlap of the samples in s. Defaults to 0.

    Returns:
        ndarray: Nd-array containing the audiosplits.
    """
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
    """Creates a torch dataloader. Optionally shuffles the data.

    Args:
        data (Dataset): The dataloader data.
        batch_size (int): Batch size.
        shuffle (bool, optional): If true schuffles data. Defaults to False.

    Returns:
        DataLoader: The torch Dataloader.
    """
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

# Spectograms
def audio_to_spectrogram(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    """Converts audio to stft spectograms and optionally converts the spectograms to log scale.

    Args:
        audio (ndarray): Audio data._
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        log (bool, optional): If true converts spectogram to log scale. Defaults to True.

    Returns:
        ndarray: An stft spectogram.
    """
    logger.light_debug("Started STFT")
    stft = librosa.stft(audio, n_fft=len_fft, hop_length=hop_length)
    spec = np.abs(stft)
    if log:
        spec = librosa.amplitude_to_db(spec)
    logger.light_debug(f"Created spectogram: {spec.shape}")
    return spec

def audio_splits_to_spectograms(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, log: bool = True) -> ndarray:
    """Converts a numpy array containing multiple samples to stft spectograms and optionally converts the spectograms to log scale.

    Args:
        audio (ndarray): Audio data._
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        log (bool, optional): If true converts spectogram to log scale. Defaults to True.

    Returns:
        ndarray: An numpy array containing the stft spectograms.
    """
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
    """Uses Griffinlim to convert a spectogram to audio.

    Args:
        spec (ndarray): An STFT spectogram.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        log (bool, optional): Set to true if spectogram is in log scale. Defaults to True.

    Returns:
        ndarray: An audio file
    """
    logger.light_debug("Started GL")
    if spec.shape[0] != len_fft // 2 + 1:
        spec = np.pad(spec, ((0, abs((len_fft // 2 + 1) - spec.shape[0])), (0, 0)), mode='constant')
    if log:
        spec = librosa.db_to_amplitude(spec)
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
    """Convenience function, does the same as normalize but is set to unnormalize to spectogram range.

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

def audio_to_mel_spectogram(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, sample_rate: int = 44100, log: bool = True, min_freq: int = 30, n_mels: int = 128) -> ndarray:
    """Creates a mel spectogram from an audio file.

    Args:
        audio (ndarray): Audio data.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sample_rate (int, optional): Sample rate. Defaults to 44100.
        log (bool, optional): If true converts spectogram to log scale. Defaults to True.
        min_freq (int, optional): Sets the minimal frequency represented by spectogram. Defaults to 30.
        n_mels (int, optional): Number of mel bins. Defaults to 128.

    Returns:
        ndarray: Mel Spectogram.
    """
    logger.light_debug("Started Mel-Spec")
    spec = librosa.feature.melspectrogram(y=audio, n_fft=len_fft, hop_length=hop_length, sr=sample_rate, fmin=min_freq, n_mels=n_mels)
    if log:
        spec = librosa.amplitude_to_db(spec)
    logger.light_debug(f"Created mel-spectogram: {spec.shape}")
    return spec

def audio_splits_to_mel_spectograms(audio: ndarray, len_fft: int = 4096, hop_length: int = 512, sample_rate: int = 44100, log: bool = True, min_freq: int = 30, n_mels: int = 128) -> ndarray:
    """Creates mel spectograms from audio samples.

    Args:
        audio (ndarray): Audio samples.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sample_rate (int, optional): Sample rate. Defaults to 44100.
        log (bool, optional): If true converts spectogram to log scale. Defaults to True.
        min_freq (int, optional): Sets the minimal frequency represented by spectogram. Defaults to 30.
        n_mels (int, optional): Number of mel bins. Defaults to 128.

    Returns:
        ndarray: Mel Spectograms.
    """
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
    """Uses Griffinlim to convert a mel-spectogram to audio.

    Args:
        spec (ndarray): An STFT spectogram.
        len_fft (int, optional): STFT FFT length. Defaults to 4096.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sample_rate (int, optional): Sample rate. Defaults to 44100.
        log (bool, optional): Set to true if spectogram is in log scale. Defaults to True.

    Returns:
        ndarray: An audio file
    """
    logger.light_debug("Started GL")
    if spec.shape[0] != len_fft // 2 + 1:
        spec = np.pad(spec, ((0, abs((len_fft // 2 + 1) - spec.shape[0])), (0, 0)), mode='constant')
    if log:
        spec = librosa.db_to_amplitude(spec)
    audio: ndarray = librosa.feature.inverse.mel_to_audio(spec, sr=sample_rate, n_fft=len_fft, hop_length=hop_length)
    audio = normalize(audio, -0.99999, 0.99999)
    logger.light_debug(f"Reconstructed audio: {audio.shape}")
    return audio

def sdr(audio: ndarray, sample_rate: int = 44100, cutoff: int = 4000) -> float:
    """
    Approximate SDR without a reference by comparing audio to a low-pass filtered version.
    
    Args:
        audio (np.ndarray): Generated waveform.
        sample_rate (int): Sampling rate in Hz.
        cutoff (float): Cutoff frequency for low-pass filter (Hz).
    
    Returns:
        float: SDR in dB.
    """
    nyquist = sample_rate / 2
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

def spectral_convergence(spectrogram: ndarray, len_fft: int = 4096, hop_length: int = 512, sample_rate: int = 44100, log: bool = True) -> float:
    """Measure spectral convergence of a generated spectrogram.

    Args:
        spectrogram (ndarray): Spectogram.
        len_fft (int, optional): STFT FFT length. Defaults to 1024.
        hop_length (int, optional): STFT hop length. Defaults to 512.
        sample_rate (int, optional): Samplerate. Defaults to 44100.
        log (bool, optional): Set to true if spectogram is in log scale. Defaults to True.

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

def visualize_spectogram(spectogram: ndarray, sample_rate: int = 44100, len_fft: int = 4096) -> None:
    """Plots a spectogram.

    Args:
        spectogram (ndarray): Spectogram.
        sample_rate (int, optional): Sample rate. Defaults to 44100.
        len_fft (int, optional): STFT FTT length. Defaults to 4096.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectogram, sr=sample_rate, n_fft=len_fft)
    plt.show()


# Torch utils
def count_parameters(model: nn.Module) -> int:
    """Counts all parameters of NN module. 

    Args:
        model (nn.Module): A torch nn.Module.
    Returns:
        int: Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Audio_Data(Dataset):
    def __init__(self, data: ndarray, labels: ndarray = None, dtype: torch.dtype = torch.float32) -> None:
        """Creates a torch dataloader. 

        Args:
            data (ndarray): Data.
            labels (ndarray, optional): If labels are not given labels are set to be the data. Defaults to None.
            dtype (torch.dtype, optional): Datatype. Defaults to torch.float32.
        """
        self.data = torch.tensor(data, dtype=dtype)
        if labels != None:
            self.labels = torch.tensor(labels, dtype=dtype) 
        else:
            self.labels = torch.tensor(data, dtype=dtype)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

