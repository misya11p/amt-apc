from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

import numpy as np

from utils import config


PATH_STYLE_VECTORS = ROOT / config.path.style_vectors
PRESETS = {
    "level1": (0., 0.9, -0.5),
    "level2": (0., 1., 0.),
    "level3": (0.5, 1.05, 0.5),
}


class Sampler:
    def __init__(self, variances=(0., 0., 0.), windows=(0.5, 0.1, 0.5)):
        self.latest = None
        self.variances = variances
        self.windows = windows

        with open(PATH_STYLE_VECTORS, "r") as f:
            style_vectors = json.load(f)
        self.style_vectors = style_vectors["style_vectors"]
        self.style_vectors = {
            key: np.array(value) for key, value in self.style_vectors.items()
        }
        self.features = style_vectors["style_features"]
        self.params = style_vectors["params"]

    def __len__(self):
        return len(self.style_vectors)

    def __getitem__(self, key):
        return self.style_vectors[key]

    def random(self):
        key = np.random.choice(list(self.style_vectors.keys()))
        return self[key]

    def get_sv(self, key_vel, key_pitch, key_onset):
        sv_vel = self[key_vel][0:8]
        sv_pitch = self[key_pitch][8:16]
        sv_onset = self[key_onset][16:24]
        sv = np.concatenate([sv_vel, sv_pitch, sv_onset]).astype(np.float32)
        return sv

    def get_feature(self, key_vel, key_pitch, key_onset):
        f_Vel, _, _ = self.features[key_vel]
        _, f_pitch, _ = self.features[key_pitch]
        _, _, f_onset = self.features[key_onset]
        return f_Vel, f_pitch, f_onset

    def sample(self, params="level2"):
        if isinstance(params, str):
            if params not in PRESETS:
                raise ValueError(f"Invalid value for 'params': {params}")
            params = PRESETS[params]

        keys_vel, keys_pitch, keys_onset = self.choices(params)
        v_vel, v_pitch, v_onset = self.variances
        sv_vel = self.summarize(keys_vel, v_vel)[0:8]
        sv_pitch = self.summarize(keys_pitch, v_pitch)[8:16]
        sv_onset = self.summarize(keys_onset, v_onset)[16:24]
        sv = np.concatenate([sv_vel, sv_pitch, sv_onset]).astype(np.float32)
        self.latest = sv
        return sv

    def choices(self, params):
        mean_vel, mean_pitch, mean_onset = params
        w_vel, w_pitch, w_onset = self.windows
        r_vel = (mean_vel - (w_vel / 2), mean_vel + (w_vel / 2))
        r_pitch = (mean_pitch - (w_pitch / 2), mean_pitch + (w_pitch / 2))
        r_onset = (mean_onset - (w_onset / 2), mean_onset + (w_onset / 2))

        keys_vel = []
        keys_pitch = []
        keys_onset = []
        for key, feature in self.features.items():
            f_vel, f_pitch, f_onset = feature
            if self._isin(f_vel, r_vel):
                keys_vel.append(key)
            if self._isin(f_pitch, r_pitch):
                keys_pitch.append(key)
            if self._isin(f_onset, r_onset):
                keys_onset.append(key)
        return keys_vel, keys_pitch, keys_onset

    @staticmethod
    def _isin(x, range):
        return range[0] <= x <= range[1]

    def summarize(self, keys, variance):
        weights = np.ones(len(keys)) + np.random.randn(len(keys)) * variance
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights)
        sv = np.zeros(24)
        for key, weight in zip(keys, weights):
            sv += self[key] * weight
        return sv
