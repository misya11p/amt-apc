import sys; sys.path.append("./")
import torch
import numpy as np

from models import Pipeline


PIPELINE = Pipeline(skip_load_model=True)
CONFIG = PIPELINE.config


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


def preprocess_feature(feature: torch.Tensor) -> torch.Tensor:
    feature = np.array(feature, dtype=np.float32)

    tmp_b = np.full([CONFIG["input"]["margin_b"], CONFIG["feature"]["n_bins"]], CONFIG["input"]["min_value"], dtype=np.float32)
    len_s = int(np.ceil(feature.shape[0] / CONFIG["input"]["num_frame"]) * CONFIG["input"]["num_frame"]) - feature.shape[0]
    tmp_f = np.full([len_s+CONFIG["input"]["margin_f"], CONFIG["feature"]["n_bins"]], CONFIG["input"]["min_value"], dtype=np.float32)

    preprocessed_feature = torch.from_numpy(np.concatenate([tmp_b, feature, tmp_f], axis=0))

    return preprocessed_feature
