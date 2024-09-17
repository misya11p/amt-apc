import json
import numpy as np


with open("data/style_vector.json") as f:
    style_vectors = json.load(f)
svs = style_vectors["style_vector"]
features = style_vectors["style_feature"]
params = style_vectors["params"]


class Sampler:
    def __init__(self, variances=(0., 0., 0.), windows=(0.5, 0.1, 0.5)):
        self.latest = None
        self.variances = variances
        self.windows = windows
        self.svs = {key: np.array(value) for key, value in svs.items()}
        self.features = features
        self.params = params

    def __len__(self):
        return len(self.svs)

    def __getitem__(self, key):
        return self.svs[key]

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

    def sample(self, params=(0., 1., 0.)):
        keys_vel, keys_pitch, keys_onset = self.choices(params)
        v_vel, v_pitch, v_onset = self.variances
        sv_vel = self.summarize(keys_vel, v_vel)[0:8]
        sv_pitch = self.summarize(keys_pitch, v_pitch)[8:16]
        sv_onset = self.summarize(keys_onset, v_onset)[16:24]
        sv = np.concatenate([sv_vel, sv_pitch, sv_onset]).astype(np.float32)
        self.latest = sv
        return sv

    def choices(self, params=(0., 1., 0.)):
        mean_vel, mean_pitch, mean_onset = params
        w_vel, w_pitch, w_onset = self.windows
        r_vel = (mean_vel - (w_vel / 2), mean_vel + (w_vel / 2))
        r_pitch = (mean_pitch - (w_pitch / 2), mean_pitch + (w_pitch / 2))
        r_onset = (mean_onset - (w_onset / 2), mean_onset + (w_onset / 2))

        keys_vel = []
        keys_pitch = []
        keys_onset = []
        for key, feature in features.items():
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
