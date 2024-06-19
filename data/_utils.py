import json
import torch
import torchaudio
import torchaudio.functional as F


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)["data"]


def load_wav(path_input):
    y, sr = torchaudio.load(path_input)
    y = torch.mean(y, dim=0)
    y = F.resample(y, sr, CONFIG["feature"]["sr"])
    return y


def wav2feature(path_input):
    y, _ = torchaudio.load(path_input)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG["feature"]["sr"],
        n_fft=CONFIG["feature"]["fft_bins"],
        win_length=CONFIG["feature"]["window_length"],
        hop_length=CONFIG["feature"]["hop_sample"],
        pad_mode=CONFIG["feature"]["pad_mode"],
        n_mels=CONFIG["feature"]["mel_bins"],
        norm="slaney",
    )
    feature, = transform(y)
    feature = (torch.log(feature + CONFIG['feature']['log_offset'])).T
    return feature
