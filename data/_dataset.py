import json

import torch
from torch.utils.data import Dataset, DataLoader

from ._utils import preprocess_feature


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)["data"]


class SyncedPianoDataset(Dataset):
    def __init__(self, path_spec, n_frames):
        self.n_frames = n_frames
        spec = torch.load(path_spec)
        spec_orig = spec[0]
        spec_pianos = spec[1:]

        spec_orig = preprocess_feature(spec_orig)
        spec_orig_split = self._split(spec_orig)

        data = []
        for spec_piano in spec_pianos:
            spec_piano = preprocess_feature(spec_piano)
            spec_piano_split = self._split(spec_piano)
            for spec_orig, spec_piano in zip(spec_orig_split, spec_piano_split):
                data.append((spec_orig, spec_piano))
            data = data[:-1]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _split(self, spec):
        specs = []
        for i in range(0, len(spec), self.n_frames):
            spec_block = (spec[i:i+CONFIG["input"]["margin_b"]+self.n_frames+CONFIG["input"]["margin_f"]]).T
            specs.append(spec_block)
        return specs


class SyncedPianos:
    def __init__(self, dir_specs, n_frames, batch_size=1):
        self.data = list(dir_specs.glob("*.pth"))
        self.n_frames = n_frames
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataset = SyncedPianoDataset(self.data[idx], self.n_frames)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return dataloader
