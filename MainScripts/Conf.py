from dataclasses import dataclass


@dataclass
class Paths:
    """Paths to the directories of the project."""
    data_path: str = "../Data"
    result_path: str = "../Results"
    model_path: str = "../Models"

class Audio:
    """Settings for audio processing and spectograms."""
    sample_rate: int = 32000
    len_fft: int = 480
    len_hop: int = 288 
    time_frame_s: int = 4
    overlap:int = 1

class Models:
    """Hyperparameters and architecture for u-net and diffusion."""
    n_starting_filters: int = 32
    n_downsamples: int = 4
    time_embed_dim: int = 128
    diffusion_timesteps: int = 1000
    batch_size: int = 16
    gradient_accum: int = 2

conf = {
    "paths": Paths(),
    "audio": Audio(),
    "model": Models()
}