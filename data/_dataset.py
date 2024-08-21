import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DIR_NAME_LABEL = "label/"
DIR_NAME_SPEC = "spec/"
with open("models/config.json", "r") as f:
    CONFIG = json.load(f)["data"]
with open("data/piano2orig.json", "r") as f:
    PIANO2ORIG = json.load(f)
with open("data/style_vector.json", "r") as f:
    STYLE_VECTOR = json.load(f)["style_vector"]
with open("data/train.txt", "r") as f:
    TRAIN_IDS = f.read().splitlines()


class PianoCoversDataset(Dataset):
    def __init__(self, dir_dataset: str):
        self.dir_dataset = Path(dir_dataset)
        self.data = list((self.dir_dataset / DIR_NAME_LABEL).glob("*.npz"))
        self.data = [
            path for path in self.data
            if self.get_id_n(path)[0] in TRAIN_IDS
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = np.load(path)
        spec, sv = self.get_spec_sv(path)

        spec = torch.from_numpy(spec).float()
        sv = torch.tensor(sv).float()
        # 修正予定
        onset = torch.from_numpy(label["onset"]).float().T
        offset = torch.from_numpy(label["offset"]).float().T
        mpe = torch.from_numpy(label["mpe"]).float().T
        velocity = torch.from_numpy(label["velocity"]).T

        return spec, sv, onset, offset, mpe, velocity

    @staticmethod
    def get_id_n(path: Path):
        split = path.stem.split("_")
        n_segment = split[-1]
        id_piano = "_".join(split[:-1])
        return id_piano, n_segment

    def get_spec_sv(self, path: Path):
        id_piano, n_segment = self.get_id_n(path)
        id_orig = PIANO2ORIG[id_piano]
        fname_orig = f"{id_orig}_{n_segment}.npy"
        path_orig = self.dir_dataset / DIR_NAME_SPEC / fname_orig
        spec = np.load(path_orig)

        sv = STYLE_VECTOR[id_piano]

        return spec, sv
