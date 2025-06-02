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
    time_frame_s: float = 4.6
    overlap:int = 1



conf = {
    "paths": Paths(),
    "audio": Audio()
}