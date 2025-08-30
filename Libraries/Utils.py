# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import optuna
# Utils
import numpy as np
from scipy.signal import firwin
from numpy import ndarray
import matplotlib.pyplot as plt
import librosa, os, logging, time, soundfile, tempfile
from typing import Callable, Optional
from collections import defaultdict
#Logging
LIGHT_DEBUG: int = 15

def light_debug(self, message, *args, **kws) -> None:
    if self.isEnabledFor(LIGHT_DEBUG):
        self._log(LIGHT_DEBUG, message, args, **kws)

logging.addLevelName(LIGHT_DEBUG, "LIGHT_DEBUG")
logging.Logger.light_debug = light_debug

logger = logging.getLogger(__name__)

class AudioData():
    def __init__(self, data: Optional[ndarray] = None, spec_data: Optional[ndarray] = None, sr: int = 32000, metadata: Optional[dict] = None) -> None:
        self.data = data
        self.spec_data = spec_data
        self.chunks = None
        self.spec_chunks = None
        self.sr = sr
        self.metadata = defaultdict(dict, metadata or {})
    
    def load_audio_file(self, path: str, mono: bool = True) -> ndarray:
        audio, current_sr = librosa.load(path, sr=None, mono=mono)
        if current_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=self.sr)
        self.data = audio
        self.metadata["source"] = path
        self.metadata["shape"] = audio.shape
        logger.light_debug(f"Loaded audio from {path} of dimensions: {audio.shape}, sr: {self.sr}")
        return audio
    
    def save_audio_file(self, path: str, norm: bool = True):
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
        self.spec_data = np.load(path)["stft"]
        self.metadata["spectrogram"]["shape"] = self.spec_data.shape
        logger.light_debug(f"Spectrogram loaded from {path} of shape: {self.spec_data.shape}")
        return self.spec_data
    
    def save_spectrogram(self, path: str) -> None:
        if self.spec_data is None:
            raise ValueError("No spectrogram data to save. Load data first.")
        np.savez_compressed(path, stft=self.spec_data)
        logger.light_debug(f"Saved spectrogram to: {path}")
    
    def audio_to_spectrogram(self, len_fft: int = 1023, hop_length: int = 256, log: bool = True) -> ndarray:
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
    
    def spectrogram_to_audio(self, len_fft: int = 1023, hop_length: int = 256, log: bool = True) -> np.ndarray:
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
        min_data: float = np.min(data)
        max_data: float = np.max(data)
        scaled_data: ndarray = (data - min_data) / (max_data - min_data)
        normalized_data: ndarray = scaled_data * (max_val - min_val) + min_val
        logger.light_debug(f"Normalized to range: [{min_val},{max_val}]")
        return normalized_data

    def normalize_filewise(self, data: ndarray, min_val: float = -1, max_val: float = 1) -> ndarray:
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
        self.data = data
    def save_training_data(self, path: str, data: (ndarray | None) = None) -> None:
        if data is None and self.data is None:
            raise ValueError("No data to save")
        data = data if data else self.data
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, data)
        logger.light_debug(f"Saved ndarray to:{path}")

    def load_training_data(self, path: str) -> ndarray:
        if not path.endswith(".npy"):
            path += ".npy"
        self.data: ndarray= np.load(path)
        logger.light_debug(f"Ndarray loaded from {path} of shape: {self.data.shape}")
        return self.data

class OS():
    def __init__(self) -> None:
        pass
    def get_filenames_from_folder(self, path: str, filetype: str = None) -> list:
        if filetype != None:
            files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(filetype)]
        else:
            files: list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        logger.light_debug(f"Got filenames {files} from {path}")
        return files
    
    def path_to_remote_path(self, path: str, is_remote: bool = False) -> bool:
        if is_remote: return path[3:]
        else: return path
    
    def del_if_exists(self, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
            logger.light_debug(f"{path} deleted")
        else:
            logger.light_debug(f"{path} could not be deleted")

class ModelData():
    def __init__(self, dataset: (Dataset | None) = None, data: (ndarray | None) = None, labels: (ndarray | None) = None) -> None:
        self.data = data
        self.labels = labels
        self.val_data, self.val_labels = None, None
        self.train_data, self.train_labels = None, None
        self.train_dataset, self.val_dataset = dataset, None
    
    def load_data_from_path(self, data_path: str, label_path: (str | None) = None, shuffle: bool = True, random_seed: int = 567) -> None:
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
        if shuffle == True:
            np.random.seed(random_seed)
            indicies: ndarray = np.arange(data.shape[0])
            np.random.shuffle(indicies)
            self.data = data[indicies]
            self.labels = labels[indicies] if labels else self.data
        else:
            self.data = data
            self.labels = labels if labels else data
    
    def create_validation_split(self, n_data_samples: (int | None) = None) -> None:
        n_samples = len(self.data)
        if n_data_samples is not None and int(n_data_samples * 0.05) + n_data_samples <= n_samples:
            n_validation_samples = int(n_data_samples * 0.05)
        else:
            n_data_samples = n_samples
            n_validation_samples = int(n_data_samples * 0.05)
            n_data_samples -= n_validation_samples
        indicies: ndarray = np.arange(self.data.shape[0])
        val_indicies = np.random.choice(indicies, size=n_validation_samples, replace = False)
        data, labels = self.data, self.labels
        self.val_data, self.val_labels = data[val_indicies], labels[val_indicies]
        self.train_data, self.train_labels = np.delete(data, val_indicies, axis=0)[:n_data_samples], np.delete(labels, val_indicies, axis=0)[:n_data_samples]
        
    def create_datasets(self, data_type: torch.dtype = torch.float32) -> tuple[Dataset, (Dataset | None)]:
        self.train_dataset = AudioDataset(self.train_data, self.train_labels, data_type=data_type)
        self.val_dataset = AudioDataset(self.val_data, self.val_labels, data_type=data_type) if self.val_data is not None else None
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, batch_size: int, shuffle: bool = False, num_workers: int = 1) -> tuple[DataLoader, (DataLoader | None)]:
        if self.train_dataset is None:
            raise ValueError("No train dataset defined")
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) if self.val_dataset else None
        return self.train_dataloader, self.val_dataloader

class AudioDataset(Dataset):
    def __init__(self, data: (ndarray | Tensor), labels: (ndarray | Tensor | None) = None, data_type: torch.dtype = torch.float32) -> None:
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

class PQMF(nn.Module):
    def __init__(self, N: int, taps: int, beta: float, device: str = "cpu" ) -> None:
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
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        assert C == 1, "PQMF.analysis expects mono input (B,1,L)."
        x_p = F.pad(x, (self.pad, self.pad), mode='reflect')
        y = F.conv1d(x_p, self.analysis_filter, stride=self.N, padding=0)
        return y

    def synthesis(self, subbands: Tensor, length: int | None = None) -> Tensor:
        w = self.synthesis_filter  # (N,1,M)
        x_p = F.conv_transpose1d(subbands, w, stride=self.N, padding=self.pad)
        if length is not None:
            x_p = x_p[..., :length]
        return x_p
    
    def reconstruct_bands(self, x: Tensor) -> Tensor:
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
        sub = self.analysis(x)
        rec = self.synthesis(sub, length=x.shape[-1])
        return sub, rec

    def _build_diag_synthesis_weight(self) -> Tensor:
        N, M = self.N, self.M
        W = torch.zeros(N, N, M, dtype=self.synthesis_filter.dtype, device=self.synthesis_filter.device)
        for i in range(N):
            W[i, i, :] = self.synthesis_filter[i, 0, :]
        return W

    @torch.no_grad()
    def synthesize_bands(self, subbands: Tensor, length: int | None = None) -> Tensor:
        if not hasattr(self, "_synth_diag") or self._synth_diag.shape[0] != self.N:
            self._synth_diag = self._build_diag_synthesis_weight()
        y = F.conv_transpose1d(subbands, self._synth_diag, stride=self.N, padding=self.pad)
        if length is not None:
            y = y[..., :length]
        return y

    @torch.no_grad()
    def synthesize_single_band(self, subbands: Tensor, k: int, length: int | None = None) -> Tensor:
        y_all = self.synthesize_bands(subbands, length=length)
        return y_all[:, k:k+1, :]

    @torch.no_grad()
    def synthesize_selected(self, subbands: Tensor, indices: list[int], length: int | None = None, reduce: bool = True) -> Tensor:
        y_all = self.synthesize_bands(subbands, length=length)
        y_sel = y_all[:, indices, :]
        if reduce:
            return y_sel.sum(dim=1, keepdim=True)
        return y_sel

    @torch.no_grad()
    def test_pqmf(self, audio_batch: Tensor, num_examples: int = 2) -> None:
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
        x_rec = bands[:,0:1,:]
        for n in range(1, self.N):
            x_rec = x_rec + bands[:,n:n+1,:]
        if length is not None:
            x_rec = x_rec[:,:,:length]
        return x_rec

    @torch.no_grad()
    def synthesize_bands(self, bands: Tensor, length: int = None) -> Tensor:
        if length is not None and bands.shape[-1] != length:
            return F.interpolate(bands, size=length, mode="linear", align_corners=False)
        return bands

    @torch.no_grad()
    def synthesize_single_band(self, bands: Tensor, k: int, length: int = None) -> Tensor:
        y_all = self.synthesize_bands(bands, length=length)
        return y_all[:, k:k+1, :]

    @torch.no_grad()
    def synthesize_selected(self, bands: Tensor, indices: list[int], length: int = None, reduce: bool = True) -> Tensor:
        y_all = self.synthesize_bands(bands, length=length)
        y_sel = y_all[:, indices, :]
        if reduce:
            return y_sel.sum(dim=1, keepdim=True)
        return y_sel

    @torch.no_grad()
    def test_filterbank(self, audio_batch: Tensor, num_examples: int = 2):
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

class TrainingUtils():
    def __init__(self) -> None:
        ...
    def random_crop_batch(self, audio: Tensor, seq_len: int) -> Tensor:
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
        return nn.functional.mse_loss(a, b).item()

    def snr_db(self, ref: Tensor, est: Tensor) -> float:
        num = torch.sum(ref.float() ** 2).item()
        err = torch.sum((ref.float() - est.float()) ** 2).item() + 1e-12
        return 10.0 * np.log10(num / err) if err > 0 else float("inf")

    def diagnostics_v_obj(self, diffusion, u_net: nn.Module, val_dataloader: DataLoader, num_samples: int = 4, len_sample: int = 2**18, sigma_list: list[float] = [0.05, 0.25, 0.5], downsample_plot: int = 16,show_spectrogram: bool = False):
        u_net.eval()
        diagnostics: dict = {}

        # Load batch
        audio_batch, _ = next(iter(val_dataloader))
        audio_batch = audio_batch.to(diffusion.device)
        if audio_batch.ndim == 2:
            audio_batch = audio_batch.unsqueeze(1)
        audio_batch = audio_batch[:num_samples, :, :len_sample].detach().clone()
        
        B, C, L = audio_batch.shape
        logger.info(f"Diagnostics on batch shape {audio_batch.shape}")

        # If filterbank exists, process audio to N bands
        if diffusion.fb is not None:
            audio_input = diffusion.fb.analysis(audio_batch)  # shape [B, N, L]
        else:
            audio_input = audio_batch  # shape [B, 1, L]

        for sigma_val in sigma_list:
            logger.info(f"\n=== Sigma = {sigma_val} ===")
            sigma_b = torch.full((B,), float(sigma_val), device=diffusion.device, dtype=torch.float32)

            # Add noise in v-objective space
            x_sigma, eps_true = diffusion.noise_img_v_obj(audio_input, sigma_b)
            a, b = diffusion.get_semicircle_weights(sigma_b)
            true_v = a * eps_true - b * audio_input

            # Reconstruct eps/x0 from true_v
            eps_from_true_v = (true_v + b * audio_input) / a
            x0_from_true_v = (a * eps_from_true_v - true_v) / b

            mse_eps_rec = self.mse(eps_from_true_v, eps_true)
            snr_eps_rec = self.snr_db(eps_true, eps_from_true_v)
            mse_x0_true = self.mse(x0_from_true_v, audio_input)
            snr_x0_true = self.snr_db(audio_input, x0_from_true_v)

            logger.info(f"eps recovery from true_v -> MSE={mse_eps_rec:.4e}, SNR={snr_eps_rec:.2f} dB")
            logger.info(f"x0 recovery from true_v -> MSE={mse_x0_true:.4e}, SNR={snr_x0_true:.2f} dB")

            # Model prediction
            with torch.no_grad():
                pred_v = u_net(x_sigma, sigma_b)

            # Compute metrics
            mse_v = self.mse(pred_v, true_v)
            snr_v = self.snr_db(true_v, pred_v)
            eps_from_pred_v = (pred_v + b * audio_input) / a
            x0_from_pred_v = (a * eps_from_pred_v - pred_v) / b

            mse_eps_pred = self.mse(eps_from_pred_v, eps_true)
            snr_eps_pred = self.snr_db(eps_true, eps_from_pred_v)
            mse_x0_pred = self.mse(x0_from_pred_v, audio_input)
            snr_x0_pred = self.snr_db(audio_input, x0_from_pred_v)

            logger.info(f"pred_v vs true_v -> MSE={mse_v:.4e}, SNR={snr_v:.2f} dB")
            logger.info(f"eps from pred_v -> MSE={mse_eps_pred:.4e}, SNR={snr_eps_pred:.2f} dB")
            logger.info(f"x0 from pred_v -> MSE={mse_x0_pred:.4e}, SNR={snr_x0_pred:.2f} dB")

            # One-step consistency
            sigma_next = max(0.0, sigma_val - 0.01)
            sigma_next_b = torch.full((B,), sigma_next, device=diffusion.device)
            a1, b1 = diffusion.get_semicircle_weights(sigma_next_b)
            x_next_true = a1 * audio_input + b1 * eps_from_true_v
            x_next_pred = a1 * x0_from_pred_v + b1 * eps_from_pred_v
            mse_step = self.mse(x_next_true, x_next_pred)
            logger.info(f"One-step next-state consistency: MSE={mse_step:.4e}")

            diagnostics[sigma_val] = {
                "mse_eps_rec_true": mse_eps_rec,
                "snr_eps_rec_true": snr_eps_rec,
                "mse_x0_true": mse_x0_true,
                "snr_x0_true": snr_x0_true,
                "mse_v": mse_v,
                "snr_v": snr_v,
                "mse_eps_pred": mse_eps_pred,
                "snr_eps_pred": snr_eps_pred,
                "mse_x0_pred": mse_x0_pred,
                "snr_x0_pred": snr_x0_pred,
                "mse_xnext": mse_step
            }

            # Plot velocities and x0 reconstructions
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
                axes[0].set_title(f"Velocities σ={sigma_val:.3f} sample={i}")
                axes[0].legend()

                axes[1].plot(xs, label="x_sigma (noisy)")
                axes[1].plot(x0t, label="x0 true")
                axes[1].plot(x0_true_rec, label="x0 from true_v", linestyle="--")
                axes[1].plot(x0_pred_rec, label="x0 from pred_v", linestyle=":")
                axes[1].set_title(f"x0 reconstructions σ={sigma_val:.3f} sample={i}")
                axes[1].legend()

                fig.tight_layout()
                plt.show()

        logger.info("\nDiagnostics complete.")
        return diagnostics



    def visualize_audio_and_spect(self, audio: ndarray) -> None:
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

def generate_latent_repr(vq_vae: nn.Module, data: Dataset, batch_size: int = 24, device: str = "cpu") -> Dataset:
    data_loader = DataLoader(data, batch_size=batch_size)
    converted_data = []
    vq_vae.eval()
    for b,_ in (data_loader):
        _, q_z, _, _ = vq_vae(b.unsqueeze(1).to(device))
        converted_data.append(q_z)
    vq_vae.train()
    new_data = torch.cat(converted_data, dim=0)
    return AudioDataset(new_data, data_type=torch.long)

class DiffusionTrainer_depr():
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
                        OS().del_if_exists(last_path)
                    logger.light_debug(f"Checkpoint saved model to {checkpoint_path}")
                continue
            break


        torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict(), "scheduler": self.lr_scheduler.state_dict(), "epoch": e + 1}, full_model_path)

        logger.light_debug(f"Saved model to {full_model_path}")

        if checkpoint_freq > 0:
            checkpoint_path: str = f"{full_model_path[:-4]}_epoch_{e + 1 - ((e + 1) % checkpoint_freq):03d}.pth"
            OS().del_if_exists(checkpoint_path)
        
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
        samples = AudioData().normalize(samples, -1, 1)
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
            visualize_spectrogram(OS().normalize(sample, -1, 1), sr=32000)
    
    def save_samples(self, samples: ndarray, file_path_name: str, sr: int = 32000, len_fft: int = 480, len_hop: int = 288) -> None:
        for i, sample in enumerate(samples):
            audio = AudioData().spectrogram_to_audio(OS().normalize(sample, -50, 50), len_fft, len_hop)
            AudioData().save_audio_file(audio, f"{file_path_name}_{i:02d}.wav", sr=sr)

