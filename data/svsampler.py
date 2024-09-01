import json
import random
import numpy as np


BINS_VEL = (-0.5, 0.5)
BINS_PITCH = (0.9, 1.1, 1.2)
BINS_ONSET = (-0.5, 0.5)

with open("data/style_vector.json") as f:
    style_vectors = json.load(f)
svs = style_vectors["style_vector"]
features = style_vectors["style_feature"]
params = style_vectors["params"]


class Sampler:
    def __init__(self):
        self.svs = svs
        self.features = features
        self.params = params

    def __getitem__(self, key):
        return self.svs[key]

    def get_feature(self, key):
        return self.features[key]

    def sample(self, levels="medium"):
        if levels == "easy":
            levels = [0, 0, 0]
        elif levels == "medium":
            levels = [1, 1, 1]
        elif levels == "hard":
            levels = [2, 2, 2]

        keys_vel, keys_pitch, keys_onset = self.choices(levels)
        key_vel = random.choice(keys_vel)
        key_pitch = random.choice(keys_pitch)
        key_onset = random.choice(keys_onset)
        sv_vel = self[key_vel][0:8]
        sv_pitch = self[key_pitch][8:16]
        sv_onset = self[key_onset][16:24]
        sv = np.concatenate([sv_vel, sv_pitch, sv_onset]).astype(np.float32)
        return sv

    def choices(self, levels):
        req_l_vel, req_l_pitch, req_l_onset = levels
        keys_vel = []
        keys_pitch = []
        keys_onset = []
        for key, feature in self.features.items():
            f_vel, f_pitch, f_onset = feature
            l_vel = np.digitize(f_vel, BINS_VEL)
            l_pitch = np.digitize(f_pitch, BINS_PITCH)
            l_onset = np.digitize(f_onset, BINS_ONSET)

            if (l_vel is None) or (l_vel == req_l_vel):
                keys_vel.append(key)
            if (l_pitch is None) or (l_pitch == req_l_pitch):
                keys_pitch.append(key)
            if (l_onset is None) or (l_onset == req_l_onset):
                keys_onset.append(key)
        return keys_vel, keys_pitch, keys_onset