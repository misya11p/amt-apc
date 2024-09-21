from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import config, info
from .sv.sampler import Sampler as SVSampler


DIR_DATASET = ROOT / config.path.dataset / "dataset/"
DIR_SPEC = DIR_DATASET / "spec/"
DIR_LABEL = DIR_DATASET / "label/"


class PianoCoversDataset(Dataset):
    def __init__(self, split="train"):
        self.data = list(DIR_LABEL.glob("*.npz"))
        if split == "train":
            self.data = [path for path in self.data if self.is_train(path)]
        elif split == "test":
            self.data = [path for path in self.data if not self.is_train(path)]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Invalid value for 'split': {split}")
        self.sv_sampler = SVSampler()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = np.load(path)
        spec, sv = self.get_spec_sv(path)

        spec = torch.from_numpy(spec).float()
        sv = torch.tensor(sv).float()
        onset = torch.from_numpy(label["onset"])
        offset = torch.from_numpy(label["offset"])
        frame = torch.from_numpy(label["frame"])
        velocity = torch.from_numpy(label["velocity"]).long()

        return spec, sv, onset, offset, frame, velocity

    @staticmethod
    def get_id_n(path: Path):
        split = path.stem.split("_")
        n_segment = split[-1]
        id_piano = "_".join(split[:-1])
        return id_piano, n_segment

    def is_train(self, path: Path):
        return info.is_train(self.get_id_n(path)[0])

    def get_spec_sv(self, path: Path):
        id_piano, n_segment = self.get_id_n(path)
        id_orig = info.piano2orig(id_piano)
        fname_orig = f"{id_orig}_{n_segment}.npy"
        path_orig = DIR_SPEC / fname_orig
        spec = np.load(path_orig)
        sv = self.sv_sampler[id_piano]
        return spec, sv
