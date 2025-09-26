##############
# Utils Library
# By Finian Landes
##############


# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# Utils
import numpy as np
from scipy.signal import firwin
from numpy import ndarray
import matplotlib.pyplot as plt
import librosa, os, logging, time, soundfile
from typing import Optional
from collections import defaultdict

##############
# Additional Logging Level between Debug and Info
##############

LIGHT_DEBUG: int = 15

def light_debug(self, message, *args, **kws) -> None:
    if self.isEnabledFor(LIGHT_DEBUG):
        self._log(LIGHT_DEBUG, message, args, **kws)

logging.addLevelName(LIGHT_DEBUG, "LIGHT_DEBUG")
logging.Logger.light_debug = light_debug
logger = logging.getLogger(__name__)

##############
# Main Util Classes
##############

class AudioData():
    def __init__(self, data: Optional[ndarray] = None, spec_data: Optional[ndarray] = None, sr: int = 32000, metadata: Optional[dict] = None) -> None:
        """Initialize the AudioData object.

        Args:
            data (Optional[ndarray], optional): The audio data. Defaults to None.
            spec_data (Optional[ndarray], optional): The spectrogram data. Defaults to None.
            sr (int, optional): The sample rate. Defaults to 32000.
            metadata (Optional[dict], optional): The metadata for the audio file. Defaults to None.
        """
        self.data = data
        self.spec_data = spec_data
        self.chunks = None
        self.spec_chunks = None
        self.sr = sr
        self.metadata = defaultdict(dict, metadata or {})
    
    def load_audio_file(self, path: str, mono: bool = True) -> ndarray:
        """Load an audio file.

        Args:
            path (str): The path to the audio file.
            mono (bool, optional): Whether to load the audio file in mono. Defaults to True.

        Returns:
            ndarray: The loaded audio data.
        """
        audio, current_sr = librosa.load(path, sr=None, mono=mono)
        if current_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=self.sr)
        self.data = audio
        self.metadata["source"] = path
        self.metadata["shape"] = audio.shape
        logger.light_debug(f"Loaded audio from {path} of dimensions: {audio.shape}, sr: {self.sr}")
        return audio
    
    def save_audio_file(self, path: str, norm: bool = True) -> None:
        """Save audio data to a file.

        Args:
            path (str): The path to the file to save the audio data.
            norm (bool, optional): Whether to normalize the audio data before saving. Defaults to True.

        Raises:
            ValueError: If no audio data is available to save.
        """

        if self.data is None:
            raise ValueError("No audio data to save. Load data first.")
        if not path.endswith((".wav", ".mp3", ".flac")):
            path += ".wav"
        audio = self.data
        if norm and audio.dtype != np.int16:
            audio = self.normalize(audio, -0.99999, 0.99999)
        soundfile.write(path, audio, self.sr)
        logger.light_debug(f"Saved audio to: {path}")
    
    def split_audiofile(self, length: float, overlap_s: float = 0, norm: bool = True) -> ndarray:
        """Split the audio file into chunks.

        Args:
            length (float): The length of each chunk in seconds.
            overlap_s (float, optional): The overlap between chunks in seconds. Defaults to 0.
            norm (bool, optional): Whether to normalize each chunk. Defaults to True.

        Returns:
            ndarray: The split audio chunks.

        Raises:
            ValueError: If no audio data is available to split.
        """
        if self.data is None:
            raise ValueError("Data missing you need to load an audiofile first")
        audio = self.data
        samples: int = int(self.sr * length)
        samples_overlap: int = int(self.sr * overlap_s)
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
        if norm:
            data = self.normalize_filewise(data)
        self.chunks = data
        self.metadata["n_chunks"] = len(data)
        self.metadata["len_chunk"] = length
        logger.light_debug(f"Split audio to: {data.shape}")
        return data

    def load_spectrogram(self, path: str) -> ndarray:
        """Load a spectrogram from a file.

        Args:
            path (str): The path to the spectrogram file.

        Returns:
            ndarray: The loaded spectrogram data.
        """
        self.spec_data = np.load(path)["stft"]
        self.metadata["spectrogram"]["shape"] = self.spec_data.shape
        logger.light_debug(f"Spectrogram loaded from {path} of shape: {self.spec_data.shape}")
        return self.spec_data
    
    def save_spectrogram(self, path: str) -> None:
        """Save the spectrogram to a file.

        Args:
            path (str): The path to the spectrogram file.

        Raises:
            ValueError: If no spectrogram data is available to save.
        """
        if self.spec_data is None:
            raise ValueError("No spectrogram data to save. Load data first.")
        np.savez_compressed(path, stft=self.spec_data)
        logger.light_debug(f"Saved spectrogram to: {path}")
    
    def audio_to_spectrogram(self, len_fft: int = 1023, hop_length: int = 256, log: bool = True) -> ndarray:
        """Convert audio to spectrogram.

        Args:
            len_fft (int, optional): The length of the FFT window. Defaults to 1023.
            hop_length (int, optional): The hop length for the STFT. Defaults to 256.
            log (bool, optional): Whether to apply log scaling. Defaults to True.

        Raises:
            ValueError: If no audio data is available to convert.

        Returns:
            ndarray: The converted spectrogram.
        """
        if self.data is None:
            raise ValueError("No audio data to convert. Load data first.")
        logger.light_debug("Started STFT")
        stft = librosa.stft(self.data, n_fft=len_fft, hop_length=hop_length)
        spec = np.abs(stft)
        if log:
            spec = np.log(spec + 1e-6)
        self.spec_data = spec
        self.metadata["spectrogram"]["shape"] = spec.shape
        logger.light_debug(f"Created spectrogram: {spec.shape}")
        return spec
    
    def audio_splits_to_spectrograms(self, len_fft: int = 1023, hop_length: int = 256, log: bool = True) -> ndarray:
        """Convert audio splits to spectrograms.

        Args:
            len_fft (int, optional): The length of the FFT window. Defaults to 1023.
            hop_length (int, optional): The hop length for the STFT. Defaults to 256.
            log (bool, optional): Whether to apply log scaling. Defaults to True.

        Raises:
            ValueError: If no audio chunks are available to convert.

        Returns:
            ndarray: The converted spectrograms.
        """
        if self.chunks is None:
            raise ValueError("No audio chunks to convert. Split audio first.")
        logger.light_debug("Started STFT on splits")
        specs = []
        for i, split in enumerate(self.chunks):
            stft = librosa.stft(split, n_fft=len_fft, hop_length=hop_length)
            spec = np.abs(stft)
            if log:
                spec = np.log(spec + 1e-6)
            specs.append(spec)
            if (i + 1) % 10 == 0 and logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - LIGHT DEBUG - Processed Splits: {i + 1}", end='')
        if logger.getEffectiveLevel() == 10:
            print()
        specs = np.array(specs)
        self.spec_chunks = specs
        self.metadata["shape"] = specs.shape
        logger.debug(f"Created spectrograms of splits: {specs.shape}")
        return specs
    
    def spectrogram_to_audio(self, len_fft: int = 1023, hop_length: int = 256, log: bool = True) -> ndarray:
        """Convert a spectrogram back to audio.

        Args:
            len_fft (int, optional): The length of the FFT window. Defaults to 1023.
            hop_length (int, optional): The hop length for the STFT. Defaults to 256.
            log (bool, optional): Whether to apply log scaling. Defaults to True.

        Raises:
            ValueError: If no spectrogram data is available to convert.

        Returns:
            ndarray: The converted audio.
        """
        if self.spec_data is None:
            raise ValueError("No spectrogram data to convert. Load or create spectrogram first.")
        logger.debug("Started GL")
        spec = self.spec_data
        if spec.shape[0] != len_fft // 2 + 1:
            spec = np.pad(spec, ((0, abs((len_fft // 2 + 1) - spec.shape[0])), (0, 0)), mode='constant')
        if log:
            spec = np.exp(spec)
        audio = librosa.griffinlim(spec, n_fft=len_fft, hop_length=hop_length)
        audio = self.normalize(audio, -0.99999, 0.99999)
        self.data = audio
        self.metadata["shape"] = audio.shape
        logger.debug(f"Reconstructed audio: {audio.shape}")
        return audio

    def audio_splits_to_mel_spectrograms(self, len_fft: int = 1023, hop_length: int = 256, min_freq: int = 30, max_freq: int = 16000, n_mels: int = 128, log: bool = True) -> ndarray:
        """Convert audio splits to mel-spectrograms.

        Args:
            len_fft (int, optional): The length of the FFT window. Defaults to 1023.
            hop_length (int, optional): The hop length for the STFT. Defaults to 256.
            min_freq (int, optional): The minimum frequency for the mel filter bank. Defaults to 30.
            max_freq (int, optional): The maximum frequency for the mel filter bank. Defaults to 16000.
            n_mels (int, optional): The number of mel bands. Defaults to 128.
            log (bool, optional): Whether to apply log scaling. Defaults to True.

        Raises:
            ValueError: If no audio chunks are available to convert.

        Returns:
            ndarray: The converted mel-spectrograms.
        """
        if self.chunks is None:
            raise ValueError("No audio chunks to convert. Split audio first.")
        logger.debug("Started Mel-Spec on splits")
        specs = []
        for i, split in enumerate(self.chunks):
            spec = librosa.feature.melspectrogram(y=split, n_fft=len_fft, hop_length=hop_length, sr=self.sr, fmin=min_freq, fmax = max_freq, n_mels=n_mels)
            if log:
                spec = np.log(spec + 1e-6)
            specs.append(spec)
            if (i + 1) % 10 == 0 and logger.getEffectiveLevel() == LIGHT_DEBUG:
                print(f"\r{time.strftime('%Y-%m-%d %H:%M:%S')},000 - lIGHT DEBUG - Processed Splits: {i + 1}", end='')
        if logger.getEffectiveLevel() == 10:
            print()
        specs = np.array(specs)
        self.spec_chunks = specs
        self.metadata["shape"] = specs.shape
        logger.debug(f"Created mel-spectrograms of splits: {specs.shape}")
        return specs
    
    def audio_to_mel_spectrogram(self, len_fft: int = 1023, hop_length: int = 256, min_freq: int = 30, max_freq: int = 16000, n_mels: int = 128, log: bool = True) -> ndarray:
        """Convert audio to mel-spectrogram.

        Args:
            len_fft (int, optional): The length of the FFT window. Defaults to 1023.
            hop_length (int, optional): The hop length for the STFT. Defaults to 256.
            min_freq (int, optional): The minimum frequency for the mel filter bank. Defaults to 30.
            max_freq (int, optional): The maximum frequency for the mel filter bank. Defaults to 16000.
            n_mels (int, optional): The number of mel bands. Defaults to 128.
            log (bool, optional): Whether to apply log scaling. Defaults to True.

        Raises:
            ValueError: If no audio data is available to convert.

        Returns:
            ndarray: The converted mel-spectrogram.
        """
        if self.data is None:
            raise ValueError("No audio data to convert. Load data first.")
        logger.debug("Started Mel-Spec")
        spec = librosa.feature.melspectrogram(y=self.data, n_fft=len_fft, hop_length=hop_length, sr=self.sr, fmin=min_freq, fmax=max_freq, n_mels=n_mels)
        if log:
            spec = np.log(spec + 1e-6)
        self.spec_data = spec
        self.metadata["shape"] = spec.shape
        logger.debug(f"Created mel-spectrogram: {spec.shape}")
        return spec
    
    def mel_spectrogram_to_audio(self, len_fft: int = 1023, hop_length: int = 256, min_freq: int = 30, max_freq: int = 16000, log: bool = True) -> ndarray:
        """Convert mel-spectrogram to audio.

        Args:
            len_fft (int, optional): The length of the FFT window. Defaults to 1023.
            hop_length (int, optional): The hop length for the STFT. Defaults to 256.
            min_freq (int, optional): The minimum frequency for the mel filter bank. Defaults to 30.
            max_freq (int, optional): The maximum frequency for the mel filter bank. Defaults to 16000.
            log (bool, optional): Whether to apply log scaling. Defaults to True.

        Raises:
            ValueError: If no spectrogram data is available to convert.

        Returns:
            ndarray: The reconstructed audio signal.
        """
        if self.spec_data is None:
            raise ValueError("No spectrogram data to convert. Load or create spectrogram first.")
        logger.debug("Started GL")
        spec = self.spec_data
        if log:
            spec = np.exp(spec)
        audio = librosa.feature.inverse.mel_to_audio(spec, sr=self.sr, n_fft=len_fft, hop_length=hop_length, fmin=min_freq, fmax=max_freq)
        audio = self.normalize(audio, -0.99999, 0.99999)
        self.data = audio
        self.metadata["shape"] = audio.shape
        logger.debug(f"Reconstructed audio: {audio.shape}")
        return audio
    
    def normalize(self, data: ndarray, min_val: float = -1, max_val: float = 1) -> ndarray:
        """Normalize the audio data to a specified range.

        Args:
            data (ndarray): The audio data to normalize.
            min_val (float, optional): The minimum value of the normalized data. Defaults to -1.
            max_val (float, optional): The maximum value of the normalized data. Defaults to 1.

        Returns:
            ndarray: The normalized audio data.
        """
        min_data: float = np.min(data)
        max_data: float = np.max(data)
        scaled_data: ndarray = (data - min_data) / (max_data - min_data)
        normalized_data: ndarray = scaled_data * (max_val - min_val) + min_val
        logger.light_debug(f"Normalized to range: [{min_val},{max_val}]")
        return normalized_data

    def normalize_filewise(self, data: ndarray, min_val: float = -1, max_val: float = 1) -> ndarray:
        """Normalize the audio data file-wise to a specified range.

        Args:
            data (ndarray): The audio data to normalize.
            min_val (float, optional): The minimum value of the normalized data. Defaults to -1.
            max_val (float, optional): The maximum value of the normalized data. Defaults to 1.

        Returns:
            ndarray: The normalized audio data.
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
    
    def __repr__(self) -> str:
        """Represent the AudioData object.

        Returns:
            str: A string representation of the AudioData object.
        """
        base = f"AudioData(sr={self.sr} Hz)"
        details = []
        
        if self.data is not None:
            shape = self.metadata.get("shape", self.data.shape if hasattr(self.data, "shape") else "N/A")
            details.append(f"audio_data(shape={shape})")
        
        if self.spec_data is not None:
            shape = self.metadata.get("spectrogram", {}).get("shape", self.spec_data.shape if hasattr(self.spec_data, "shape") else "N/A")
            details.append(f"spectrogram_data(shape={shape})")
        
        if self.chunks is not None:
            n_chunks = self.metadata.get("n_chunks", len(self.chunks) if hasattr(self.chunks, "__len__") else "N/A")
            len_chunk = self.metadata.get("len_chunk", "N/A")
            details.append(f"audio_chunks(n_chunks={n_chunks}, len_chunk={len_chunk}s)")
        
        if self.spec_chunks is not None:
            shape = self.metadata.get("shape", self.spec_chunks.shape if hasattr(self.spec_chunks, "shape") else "N/A")
            details.append(f"spectrogram_chunks(shape={shape})")
        
        if not details:
            return f"{base}: No data loaded"
        
        return f"{base}: {', '.join(details)}"

class NPData():
    def __init__(self, data: (ndarray | None) = None) -> None:
        """Numpy Array Data Container

        Args:
            data (ndarray  |  None, optional): Numpy array data. Defaults to None.
        """
        self.data = data
    def save_training_data(self, path: str, data: (ndarray | None) = None) -> None:
        """Save training data to a file.

        Args:
            path (str): The path to the file to save the data.
            data (ndarray  |  None, optional): The data to save. Defaults to None.

        Raises:
            ValueError: If no data is available to save.
        """
        if data is None and self.data is None:
            raise ValueError("No data to save")
        data = data if data else self.data
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, data)
        logger.light_debug(f"Saved ndarray to:{path}")

    def load_training_data(self, path: str) -> ndarray:
        """
        Load training data from a file.

        Args:
            path (str): The path to the file to load the data from.

        Returns:
            ndarray: The loaded data.
        """
        if not path.endswith(".npy"):
            path += ".npy"
        self.data: ndarray= np.load(path)
        logger.light_debug(f"Ndarray loaded from {path} of shape: {self.data.shape}")
        return self.data

class OS():
    def __init__(self) -> None:
        """Operating System Utilities"""
        ...
    def get_filenames_from_folder(self, path: str, filetype: str = None) -> list:
        """Get a list of filenames from a folder.

        Args:
            path (str): The path to the folder.
            filetype (str, optional): The file extension to filter by. Defaults to None.

        Returns:
            list: A list of filenames in the folder.
        """
        if filetype != None:
            files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(filetype)]
        else:
            files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        logger.light_debug(f"Got filenames {files} from {path}")
        return files
    
    def path_to_remote_path(self, path: str, is_remote: bool = False) -> bool:
        """Convert a local path to a remote path .

        Args:
            path (str): The local path to convert.
            is_remote (bool, optional): Whether the path is already a remote path. Defaults to False.

        Returns:
            bool: True if the path was converted to a remote path, False otherwise.
        """
        if is_remote: return path[3:]
        else: return path
    
    def del_if_exists(self, path: str) -> None:
        """Delete a file if it exists.

        Args:
            path (str): The path to the file to delete.
        """
        if os.path.exists(path):
            os.remove(path)
            logger.light_debug(f"{path} deleted")
        else:
            logger.light_debug(f"{path} could not be deleted")

class ModelData():
    def __init__(self, dataset: (Dataset | None) = None, data: (ndarray | None) = None, labels: (ndarray | None) = None) -> None:
        """Creates a ModelData instance for managing audio data and labels.

        Args:
            dataset (Dataset  |  None, optional): The dataset object containing audio data. Defaults to None.
            data (ndarray  |  None, optional): The audio data array. Defaults to None.
            labels (ndarray  |  None, optional): The corresponding labels for the audio data. Defaults to None.
        """
        self.data = data
        self.labels = labels
        self.val_data, self.val_labels = None, None
        self.train_data, self.train_labels = None, None
        self.train_dataset, self.val_dataset = dataset, None
    
    def load_data_from_path(self, data_path: str, label_path: (str | None) = None, shuffle: bool = True, random_seed: int = 567) -> None:
        """Loads audio data and labels from the specified file paths.

        Args:
            data_path (str): The file path to the audio data.
            label_path (str  |  None, optional): The file path to the audio labels. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            random_seed (int, optional): The random seed for shuffling. Defaults to 567.
        """
        data = NPData().load_training_data(data_path)
        labels = NPData().load_training_data(label_path) if label_path else None
        if shuffle == True:
            np.random.seed(random_seed)
            indicies: ndarray = np.arange(data.shape[0])
            np.random.shuffle(indicies)
            self.data = data[indicies]
            self.labels = labels[indicies] if labels else self.data
        else:
            self.data = data
            self.labels = labels if labels else data

    def load_data(self, data: ndarray, labels: (ndarray | None) = None, shuffle: bool = True, random_seed: int = 567) -> None:
        """Loads audio data and labels into the ModelData instance.

        Args:
            data (ndarray): The audio data array.
            labels (ndarray | None, optional): The corresponding labels for the audio data. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            random_seed (int, optional): The random seed for shuffling. Defaults to 567.
        """
        if shuffle == True:
            np.random.seed(random_seed)
            indicies: ndarray = np.arange(data.shape[0])
            np.random.shuffle(indicies)
            self.data = data[indicies]
            self.labels = labels[indicies] if labels else self.data
        else:
            self.data = data
            self.labels = labels if labels else data

    def create_validation_split(self, n_data_samples: int | None = None) -> None:
        """Creates a validation split from the training data.

        Args:
            n_data_samples (int | None, optional): The number of training samples to use.
                If None, uses the full dataset. Validation is always 5% of chosen data.
        """
        n_samples = len(self.data)

        if n_data_samples is None or n_data_samples > n_samples:
            n_data_samples = n_samples

        n_validation_samples = max(1, int(n_data_samples * 0.05))
        n_train_samples = n_data_samples - n_validation_samples

        indices = np.arange(n_samples)
        val_indices = np.random.choice(indices, size=n_validation_samples, replace=False)
        train_indices = np.setdiff1d(indices, val_indices)[:n_train_samples]

        self.val_data, self.val_labels = self.data[val_indices], self.labels[val_indices]
        self.train_data, self.train_labels = self.data[train_indices], self.labels[train_indices]

        
    def create_datasets(self, data_type: torch.dtype = torch.float32) -> tuple[Dataset, (Dataset | None)]:
        """Creates the training and validation datasets.

        Args:
            data_type (torch.dtype, optional): The data type for the audio tensors. Defaults to torch.float32.

        Returns:
            tuple: A pair (train_dataset, validation_dataset).
                The second element may be None if no validation set is created.

        """
        self.train_dataset = AudioDataset(self.train_data, self.train_labels, data_type=data_type)
        self.val_dataset = AudioDataset(self.val_data, self.val_labels, data_type=data_type) if self.val_data is not None else None
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, batch_size: int, shuffle: bool = False, num_workers: int = 1) -> tuple[DataLoader, (DataLoader | None)]:
        """Creates the training and validation dataloaders.

        Args:
            batch_size (int): The batch size for the dataloaders.
            shuffle (bool, optional): Whether to shuffle the training data. Defaults to False.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 1.

        Returns:
            tuple: A pair (train_dataloader, validation_dataloader).
                    The second element may be None if no validation set is created.

        Raises:
            ValueError: If no train dataset is defined.
        """
        if self.train_dataset is None:
            raise ValueError("No train dataset defined")
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) if self.val_dataset else None
        return self.train_dataloader, self.val_dataloader

class AudioDataset(Dataset):
    def __init__(self, data: (ndarray | Tensor), labels: (ndarray | Tensor | None) = None, data_type: torch.dtype = torch.float32) -> None:
        """Creates a Dataset for audio data.

        Args:
            data (ndarray  |  Tensor): The audio data.
            labels (ndarray  |  Tensor  |  None, optional): The corresponding labels for the audio data. Defaults to None.
            data_type (torch.dtype, optional): The data type for the audio tensors. Defaults to torch.float32.
        """
        if type(data) is not  Tensor:
            data: Tensor = torch.tensor(data)
        if type(labels) is not Tensor and labels is not None:
            labels: Tensor = torch.tensor(labels)
        if labels is not None:
            self.labels = labels.to(dtype=data_type) 
        else:
            self.labels = data.to(dtype=data_type) 
        self.data = data.to(dtype=data_type)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TrainingUtils():
    def __init__(self) -> None:
        ...
    def random_crop_batch(self, audio: Tensor, seq_len: int) -> Tensor:
        """Randomly crop each item in abatch of audio tensors to a specified sequence length.

        Args:
            audio (Tensor): The input audio tensor.
            seq_len (int): The desired sequence length.

        Returns:
            Tensor: The cropped audio tensor.
        """
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        B, C, L = audio.shape

        assert L >= seq_len, "Audio length needs to be equal or larger than the seq lenght"

        start_idx = torch.randint(0, L - seq_len + 1, (B, 1), device=audio.device)
        offsets = torch.arange(seq_len, device=audio.device).unsqueeze(0)
        indices = start_idx + offsets
        indices = indices.unsqueeze(1).expand(-1, C, -1)
        return torch.gather(audio, 2, indices)
    
    def mse(self, a: Tensor, b: Tensor) -> float:
        """Compute the Mean Squared Error (MSE) between two tensors.

        Args:
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.

        Returns:
            float: The MSE between the two tensors.
        """
        return nn.functional.mse_loss(a, b).item()

    def reprod_quality_db(self, ref: Tensor, est: Tensor) -> float:
        """Compute the Signal-to-Noise Ratio (reprod_quality) in dB.

        Args:
            ref (Tensor): The reference (clean) audio signal.
            est (Tensor): The estimated (noisy) audio signal.

        Returns:
            float: The reprod_quality in dB.
        """

        num = torch.sum(ref.float() ** 2).item()
        err = torch.sum((ref.float() - est.float()) ** 2).item() + 1e-12
        return 10.0 * np.log10(num / err) if err > 0 else float("inf")

    def diagnostics_v_obj(self, diffusion, u_net: nn.Module, val_dataloader: DataLoader, num_samples: int = 4, len_sample: int = 2**18, sigma_list: list[float] = [0.05, 0.25, 0.5], downsample_plot: int = 16) -> dict:
        """Visualize and Analyze the denoising process of the model.

        Args:
            diffusion (_type_): The Diffusion Model.
            u_net (nn.Module): The U-Net model for denoising.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            num_samples (int, optional): Number of samples to visualize. Defaults to 4.
            len_sample (int, optional): Length of the audio samples. Defaults to 2**18.
            sigma_list (list[float], optional): List of noise levels to analyze. Defaults to [0.05, 0.25, 0.5].
            downsample_plot (int, optional): Factor by which to downsample the plot. Defaults to 16.
        Returns:
            dict: A dictionary containing the diagnostic results.
        """
        u_net.eval()
        diagnostics: dict = {}

        audio_batch, _ = next(iter(val_dataloader))
        audio_batch = audio_batch.to(diffusion.device)
        if audio_batch.ndim == 2:
            audio_batch = audio_batch.unsqueeze(1)
        audio_batch = audio_batch[:num_samples, :, :len_sample].detach().clone()
        
        B, C, L = audio_batch.shape
        logger.info(f"Diagnostics on batch shape {audio_batch.shape}")

        if diffusion.fb is not None:
            audio_input = diffusion.fb.analysis(audio_batch)
        else:
            audio_input = audio_batch

        for sigma_val in sigma_list:
            logger.info(f"\n=== Sigma = {sigma_val} ===")
            sigma_b = torch.full((B,), float(sigma_val), device=diffusion.device, dtype=torch.float32)

            x_sigma, eps_true = diffusion.noise_img_v_obj(audio_input, sigma_b)
            a, b = diffusion.get_semicircle_weights(sigma_b)
            true_v = a * eps_true - b * audio_input

            eps_from_true_v = (true_v + b * audio_input) / a
            x0_from_true_v = (a * eps_from_true_v - true_v) / b

            mse_eps_rec = self.mse(eps_from_true_v, eps_true)
            reprod_quality_eps_rec = self.reprod_quality_db(eps_true, eps_from_true_v)
            mse_x0_true = self.mse(x0_from_true_v, audio_input)
            reprod_quality_x0_true = self.reprod_quality_db(audio_input, x0_from_true_v)

            logger.info(f"eps recovery from true_v -> MSE={mse_eps_rec:.4e}, reprod_quality={reprod_quality_eps_rec:.2f} dB")
            logger.info(f"x0 recovery from true_v -> MSE={mse_x0_true:.4e}, reprod_quality={reprod_quality_x0_true:.2f} dB")

            with torch.no_grad():
                pred_v = u_net(x_sigma, sigma_b)

            mse_v = self.mse(pred_v, true_v)
            reprod_quality_v = self.reprod_quality_db(true_v, pred_v)
            eps_from_pred_v = (pred_v + b * audio_input) / a
            x0_from_pred_v = (a * eps_from_pred_v - pred_v) / b

            mse_eps_pred = self.mse(eps_from_pred_v, eps_true)
            reprod_quality_eps_pred = self.reprod_quality_db(eps_true, eps_from_pred_v)
            mse_x0_pred = self.mse(x0_from_pred_v, audio_input)
            reprod_quality_x0_pred = self.reprod_quality_db(audio_input, x0_from_pred_v)

            logger.info(f"pred_v vs true_v -> MSE={mse_v:.4e}, reprod_quality={reprod_quality_v:.2f} dB")
            logger.info(f"eps from pred_v -> MSE={mse_eps_pred:.4e}, reprod_quality={reprod_quality_eps_pred:.2f} dB")
            logger.info(f"x0 from pred_v -> MSE={mse_x0_pred:.4e}, reprod_quality={reprod_quality_x0_pred:.2f} dB")

            sigma_next = max(0.0, sigma_val - 0.01)
            sigma_next_b = torch.full((B,), sigma_next, device=diffusion.device)
            a1, b1 = diffusion.get_semicircle_weights(sigma_next_b)
            x_next_true = a1 * audio_input + b1 * eps_from_true_v
            x_next_pred = a1 * x0_from_pred_v + b1 * eps_from_pred_v
            mse_step = self.mse(x_next_true, x_next_pred)
            logger.info(f"One-step next-state consistency: MSE={mse_step:.4e}")

            diagnostics[sigma_val] = {
                "mse_eps_rec_true": mse_eps_rec,
                "reprod_quality_eps_rec_true": reprod_quality_eps_rec,
                "mse_x0_true": mse_x0_true,
                "reprod_quality_x0_true": reprod_quality_x0_true,
                "mse_v": mse_v,
                "reprod_quality_v": reprod_quality_v,
                "mse_eps_pred": mse_eps_pred,
                "reprod_quality_eps_pred": reprod_quality_eps_pred,
                "mse_x0_pred": mse_x0_pred,
                "reprod_quality_x0_pred": reprod_quality_x0_pred,
                "mse_xnext": mse_step
            }

            for i in range(min(B, num_samples)):
                tv = true_v[i, 0].cpu().numpy()[::downsample_plot]
                pv = pred_v[i, 0].cpu().numpy()[::downsample_plot]
                xs = x_sigma[i, 0].cpu().numpy()[::downsample_plot]
                x0t = audio_input[i, 0].cpu().numpy()[::downsample_plot]
                x0_true_rec = x0_from_true_v[i, 0].cpu().numpy()[::downsample_plot]
                x0_pred_rec = x0_from_pred_v[i, 0].cpu().numpy()[::downsample_plot]

                fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
                axes[0].plot(tv, label="true_v")
                axes[0].plot(pv, label="pred_v", alpha=0.8)
                axes[0].set_title(f"Velocities sigma={sigma_val:.3f} sample={i}")
                axes[0].legend()

                axes[1].plot(xs, label="x_sigma (noisy)")
                axes[1].plot(x0t, label="x0 true")
                axes[1].plot(x0_true_rec, label="x0 from true_v", linestyle="--")
                axes[1].plot(x0_pred_rec, label="x0 from pred_v", linestyle=":")
                axes[1].set_title(f"x0 reconstructions sigma={sigma_val:.3f} sample={i}")
                axes[1].legend()

                fig.tight_layout()
                plt.show()

        logger.info("\nDiagnostics complete.")
        return diagnostics

    def visualize_audio_and_spect(self, audio: ndarray) -> None:
        """Visualize audio waveform and spectrogram.

        Args:
            audio (ndarray): Input audio array.
        """
        if audio.ndim == 3:
            audio = audio[0][0]
        elif audio.ndim == 2:
            audio = audio[0]
        ad = AudioData(audio)
        spect = ad.audio_to_spectrogram()
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(audio)
        axes[0].set_title("Waveform")
        axes[0].set_xlabel("Samples")
        axes[0].set_ylabel("Amplitude")

        im = axes[1].imshow(spect, aspect='auto', origin='lower', interpolation='none', cmap='magma')
        axes[1].set_title("Spectrogram")
        axes[1].set_xlabel("Time bins")
        axes[1].set_ylabel("Frequency bins")
        fig.colorbar(im, ax=axes[1], format='%+2.0f dB')

        plt.tight_layout()
        plt.show()

    def count_params(self, model: nn.Module) -> str:
        """Counts all parameters of NN module. 
        Args:
            model (nn.Module, optional): A torch nn.Module.
        Returns:
            str: Number of parameters, rounded and with suffix eg. ~5.34M.
        """
        suffixes: dict = {1e9:"B", 1e6:"M", 1e3:"k", 1e0:""}
        n =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        for key, val in suffixes.items():
            if n / key > 1:
                n = round(n / key, 3)
                return f"~{str(n)[:5]}{val}"
##############
# Additional nn.Modules
##############

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(256, 512, 128), win_lengths=(1024, 2048, 512)):
        """Multi-resolution STFT loss for comparing audio signals.

        Args:
            fft_sizes (tuple, optional): FFT sizes for STFT. Defaults to (1024, 2048, 512).
            hop_sizes (tuple, optional): Hop sizes for STFT. Defaults to (256, 512, 128).
            win_lengths (tuple, optional): Window lengths for STFT. Defaults to (1024, 2048, 512).
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass for the multi-resolution STFT loss.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Target tensor.

        Returns:
            Tensor: Computed loss.
        """
        loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            X = torch.stft(x.squeeze(1), n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=torch.hann_window(win_length, device=x.device), return_complex=True)
            Y = torch.stft(y.squeeze(1), n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=torch.hann_window(win_length, device=y.device), return_complex=True)

            mag_X = torch.abs(X)
            mag_Y = torch.abs(Y)

            sc_loss = torch.norm(mag_X - mag_Y, p=1) / torch.norm(mag_Y, p=1)
            mag_loss = F.l1_loss(mag_X, mag_Y)

            loss += sc_loss + mag_loss

        return loss / len(self.fft_sizes)

class PQMF(nn.Module):
    """Pseudo-Quadrature Mirror Filter (PQMF) implementation."""
    def __init__(self, N: int, taps: int, beta: float, device: str = "cpu" ) -> None:
        """Pseudo-Quadrature Mirror Filter (PQMF) initialization.

        Args:
            N (int): Number of subbands.
            taps (int): Number of filter taps.
            beta (float): Kaiser window beta parameter.
            device (str, optional): Device to run the model on. Defaults to "cpu".
        """
        super().__init__()
        self.N = N
        self.M = taps if taps % 2 == 0 else taps + 1
        self.pad = self.M // 2
        self.device = device

        cutoff = 1.0 / N
        t = np.arange(-self.M // 2, self.M // 2, dtype=np.float64)
        h_proto = np.sinc( 2* cutoff * t)
        win = np.kaiser(self.M, beta=beta).astype(np.float64)
        h_proto *= win
        h_proto /= np.sum(h_proto)

        h = np.zeros((N, self.M), dtype=np.float64)
        M_center = (self.M - 1) / 2.0
        for k in range(N):
            phase = (np.pi * (2 * k + 1) / (2.0 * N)) * (np.arange(self.M) - M_center)
            h[k, :] = 2.0 * h_proto * np.cos(phase + ((-1)**k) * (np.pi / 4.0))

        h_torch = torch.tensor(h).unsqueeze(1).to(device=self.device, dtype=torch.float32)

        self.register_buffer("analysis_filter", h_torch)
        self.register_buffer("synthesis_filter", h_torch.clone())

    def analysis(self, x: Tensor) -> Tensor:
        """Applies the analysis filter bank to the input signal.

        Args:
            x (Tensor): Input tensor of shape (B, 1, L).

        Returns:
            Tensor: Output tensor of shape (B, N, L').
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        assert C == 1, "PQMF.analysis expects mono input (B,1,L)."
        x_p = F.pad(x, (self.pad, self.pad), mode='reflect')
        y = F.conv1d(x_p, self.analysis_filter, stride=self.N, padding=0)
        return y

    def synthesis(self, subbands: Tensor, length: int | None = None) -> Tensor:
        """Synthesizes the full-band signal from subband signals.

        Args:
            subbands (Tensor): Input tensor of shape (B, N, L').
            length (int | None, optional): Output length. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, 1, L).
        """
        w = self.synthesis_filter
        x_p = F.conv_transpose1d(subbands, w, stride=self.N, padding=self.pad)
        if length is not None:
            x_p = x_p[..., :length]
        return x_p
    
    def reconstruct_bands(self, x: Tensor) -> Tensor:
        """Reconstructs individual band signals from the full-band input.

        Args:
            x (Tensor): Input tensor of shape (B, 1, L).

        Returns:
            Tensor: Output tensor of shape (B, N, L').
        """
        subbands = self.analysis(x)
        B, N, _ = subbands.shape
        L = x.shape[-1]
        band_signals = []
        for k in range(N):
            mask = torch.zeros_like(subbands)
            mask[:, k, :] = subbands[:, k, :]
            band_k = self.synthesis(mask, length=L)
            band_signals.append(band_k.squeeze(1))
        return torch.stack(band_signals, dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Convenience method for forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, 1, L).

        Returns:
            tuple[Tensor, Tensor]: Tuple containing the subband signals and the reconstructed signal.
        """
        sub = self.analysis(x)
        rec = self.synthesis(sub, length=x.shape[-1])
        return sub, rec

    def _build_diag_synthesis_weight(self) -> Tensor:
        """Builds the diagonal synthesis weight tensor.

        Returns:
            Tensor: Diagonal synthesis weight tensor of shape (N, N, M).
        """
        N, M = self.N, self.M
        W = torch.zeros(N, N, M, dtype=self.synthesis_filter.dtype, device=self.synthesis_filter.device)
        for i in range(N):
            W[i, i, :] = self.synthesis_filter[i, 0, :]
        return W

    @torch.no_grad()
    def synthesize_bands(self, subbands: Tensor, length: int | None = None) -> Tensor:
        """Synthesizes the full-band signal from subband signals.

        Args:
            subbands (Tensor): Input tensor of shape (B, N, L').
            length (int | None, optional): Output length. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, 1, L).
        """
        if not hasattr(self, "_synth_diag") or self._synth_diag.shape[0] != self.N:
            self._synth_diag = self._build_diag_synthesis_weight()
        y = F.conv_transpose1d(subbands, self._synth_diag, stride=self.N, padding=self.pad)
        if length is not None:
            y = y[..., :length]
        return y

    @torch.no_grad()
    def synthesize_single_band(self, subbands: Tensor, k: int, length: int | None = None) -> Tensor:
        """Synthesizes a single band from the subband signals.

        Args:
            subbands (Tensor): Input tensor of shape (B, N, L').
            k (int): Index of the band to synthesize.
            length (int | None, optional): Output length. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, 1, L).
        """
        y_all = self.synthesize_bands(subbands, length=length)
        return y_all[:, k:k+1, :]

    @torch.no_grad()
    def synthesize_selected(self, subbands: Tensor, indices: list[int], length: int | None = None, reduce: bool = True) -> Tensor:
        """Synthesizes selected bands from the subband signals.

        Args:
            subbands (Tensor): Input tensor of shape (B, N, L').
            indices (list[int]): List of band indices to synthesize.
            length (int | None, optional): Output length. Defaults to None.
            reduce (bool, optional): Whether to reduce the output. Defaults to True.

        Returns:
            Tensor: Output tensor of shape (B, 1, L) if reduce is True, else (B, len(indices), L).
        """
        y_all = self.synthesize_bands(subbands, length=length)
        y_sel = y_all[:, indices, :]
        if reduce:
            return y_sel.sum(dim=1, keepdim=True)
        return y_sel

    @torch.no_grad()
    def test_pqmf(self, audio_batch: Tensor, num_examples: int = 2) -> None:
        """Tests the PQMF layer.

        Args:
            audio_batch (Tensor): Input audio batch of shape (B, 1, L).
            num_examples (int, optional): Number of examples to visualize. Defaults to 2.
        """
        logger.info(f"Input batch: {audio_batch.shape}")

        subbands = self.analysis(audio_batch)
        logger.info(f"Subbands shape: {subbands.shape}")

        recon = self.synthesis(subbands, length=audio_batch.shape[-1])
        logger.info(f"Reconstructed shape: {recon.shape}")

        bands_up = self.synthesize_bands(subbands, length=audio_batch.shape[-1])
        logger.info(f"Bands upsampled: {bands_up.shape}")

        for i in range(min(num_examples, audio_batch.shape[0])):
            plt.figure(figsize=(14,6))

            plt.subplot(2,1,1)
            plt.plot(audio_batch[i,0].cpu().numpy(), label="original", alpha=0.7)
            plt.plot(recon[i,0].cpu().numpy(), label="reconstructed", alpha=0.7)
            plt.title(f"PQMF reconstruction check (sample {i})")
            plt.legend()

            plt.subplot(2,1,2)
            err = (audio_batch[i,0] - recon[i,0]).cpu().numpy()
            plt.plot(err)
            plt.title("Reconstruction error")
            plt.tight_layout()
            plt.show()

        for i in range(min(num_examples, audio_batch.shape[0])):
            fig, axes = plt.subplots(self.N, 1, figsize=(14, 2*self.N), sharex=True)
            fig.suptitle(f"Upsampled bands (sample {i})")

            for k in range(self.N):
                axes[k].plot(bands_up[i,k].cpu().numpy())
                axes[k].set_ylabel(f"Band {k}")
            plt.tight_layout()
            plt.show()

        sum_bands = bands_up.sum(dim=1, keepdim=True)
        diff = (recon - sum_bands).abs().max().item()
        logger.info(f"Max |recon - sum(bands)| = {diff:.2e}")

class Filterbank(nn.Module):
    def __init__(self, freq_edges: list, taps: int, sample_rate: int) -> None:
        """Filterbank initialization.

        Args:
            freq_edges (list): List of frequency edges for the filterbank.
            taps (int): Number of taps for the FIR filters.
            sample_rate (int): Sample rate of the input audio.
        """
        super().__init__()
        self.freq_edges = freq_edges
        self.taps = taps
        self.sr = sample_rate
        self.N = len(freq_edges) - 1

        filters = []
        nyq = sample_rate / 2
        eps = 1e-6
        for i in range(self.N):
            low = max(freq_edges[i] / nyq, eps)
            high = min(freq_edges[i+1] / nyq, 1 - eps)

            if i == 0:
                h = firwin(taps, high, pass_zero=True, window=("kaiser", 10))
            else:
                h = firwin(taps, [low, high], pass_zero=False, window=("kaiser", 10))

            h_tensor = torch.tensor(h, dtype=torch.float32).view(1,1,-1)
            filters.append(h_tensor)
        self.register_buffer("filters", torch.cat(filters, dim=0))  # (N,1,taps)
        self.pad = taps // 2

    def analysis(self, x: Tensor) -> Tensor:
        """Performs analysis on the input audio tensor.

        Args:
            x (Tensor): Input audio tensor of shape (B, 1, L).

        Returns:
            Tensor: Audio bands tensor of shape (B, N, L).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        bands = []
        for n in range(self.N):
            h = self.filters[n:n+1]
            y = F.conv1d(x, h, padding=self.pad)
            y = y[..., :L]
            bands.append(y)
        return torch.cat(bands, dim=1)

    def synthesis(self, bands: Tensor, length: int = None) -> Tensor:
        """Synthesizes the audio from the given bands.

        Args:
            bands (Tensor): Audio bands tensor of shape (B, N, L).
            length (int, optional): Length of the output tensor. Defaults to None.

        Returns:
            Tensor: Synthesized audio tensor of shape (B, 1, L) or (B, 1, length) if length is specified.
        """
        x_rec = bands[:,0:1,:]
        for n in range(1, self.N):
            x_rec = x_rec + bands[:,n:n+1,:]
        if length is not None:
            x_rec = x_rec[:,:,:length]
        return x_rec

    @torch.no_grad()
    def synthesize_bands(self, bands: Tensor, length: int = None) -> Tensor:
        """Synthesizes the audio from the given bands.

        Args:
            bands (Tensor): Audio bands tensor of shape (B, N, L).
            length (int, optional): Length of the output tensor. Defaults to None.

        Returns:
            Tensor: Synthesized audio tensor of shape (B, 1, L) or (B, 1, length) if length is specified.
        """
        if length is not None and bands.shape[-1] != length:
            return F.interpolate(bands, size=length, mode="linear", align_corners=False)
        return bands

    @torch.no_grad()
    def synthesize_single_band(self, bands: Tensor, k: int, length: int = None) -> Tensor:
        """Synthesizes a single audio band.

        Args:
            bands (Tensor): Audio bands tensor of shape (B, N, L).
            k (int): Index of the band to synthesize.
            length (int, optional): Length of the output tensor. Defaults to None.

        Returns:
            Tensor: Synthesized audio tensor of shape (B, 1, L) or (B, 1, length) if length is specified.
        """
        y_all = self.synthesize_bands(bands, length=length)
        return y_all[:, k:k+1, :]

    @torch.no_grad()
    def synthesize_selected(self, bands: Tensor, indices: list[int], length: int = None, reduce: bool = True) -> Tensor:
        """Synthesizes the selected audio bands.

        Args:
            bands (Tensor): Audio bands tensor of shape (B, N, L).
            indices (list[int]): List of indices of the bands to synthesize.
            length (int, optional): Length of the output tensor. Defaults to None.
            reduce (bool, optional): Whether to reduce the output tensor by summing the selected bands. Defaults to True.

        Returns:
            Tensor: Synthesized audio tensor of shape (B, 1, L) or (B, 1, length) if length is specified.
        """
        y_all = self.synthesize_bands(bands, length=length)
        y_sel = y_all[:, indices, :]
        if reduce:
            return y_sel.sum(dim=1, keepdim=True)
        return y_sel

    @torch.no_grad()
    def test_filterbank(self, audio_batch: Tensor, num_examples: int = 2) -> None:
        """Tests the filter bank by performing analysis and synthesis on the input audio.

        Args:
            audio_batch (Tensor): Input audio batch of shape (B, 1, L).
            num_examples (int, optional): Number of examples to test. Defaults to 2.
        """
        subbands = self.analysis(audio_batch)
        recon = self.synthesis(subbands, length=audio_batch.shape[-1])
        bands_up = self.synthesize_bands(subbands, length=audio_batch.shape[-1])

        for i in range(min(num_examples, audio_batch.shape[0])):
            plt.figure(figsize=(14,6))
            plt.subplot(2,1,1)
            plt.plot(audio_batch[i,0].cpu().numpy(), label="original", alpha=0.7)
            plt.plot(recon[i,0].cpu().numpy(), label="reconstructed", alpha=0.7)
            plt.title(f"Reconstruction check (sample {i})")
            plt.legend()

            plt.subplot(2,1,2)
            err = (audio_batch[i,0] - recon[i,0]).cpu().numpy()
            plt.plot(err)
            plt.title("Reconstruction error")
            plt.tight_layout()
            plt.show()

        for i in range(min(num_examples, audio_batch.shape[0])):
            fig, axes = plt.subplots(self.N, 1, figsize=(14, 2*self.N), sharex=True)
            fig.suptitle(f"Upsampled bands (sample {i})")
            for k in range(self.N):
                axes[k].plot(bands_up[i,k].cpu().numpy())
                axes[k].set_ylabel(f"Band {k}")
            plt.tight_layout()
            plt.show()

        sum_bands = bands_up.sum(dim=1, keepdim=True)
        diff = (recon - sum_bands).abs().max().item()
        logger.info(f"Max |recon - sum(bands)| = {diff:.2e}")
