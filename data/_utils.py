import torch
from models import Pipeline


PIPELINE = Pipeline(skip_load_model=True)


def wav2feature(path_input: str) -> torch.Tensor:
    """
    Convert a wav file to a feature:
    mel-spectrogram according to models/config.json

    Args:
        path_input (str): Path to the input wav file.

    Returns:
        torch.Tensor: Feature tensor. (n_frames, n_mels)
    """
    return PIPELINE.wav2feature(path_input)
