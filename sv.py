import argparse
from pathlib import Path
import json

import numpy as np
import librosa
from tqdm import tqdm


DIR_NAME_ARRAY = "array/"
DIR_NAME_SYNCED = "synced/"
DIR_NAME_PIANO = "piano/"

with open("models/config.json", "r") as f:
    CONFIG = json.load(f)["data"]
PITCH_MIN = CONFIG["midi"]["note_min"]
PITCH_MAX = CONFIG["midi"]["note_max"]
N_VELOCITY = CONFIG["midi"]["num_velocity"]
SR = CONFIG["feature"]["sr"]
HOP_LENGTH = CONFIG["feature"]["hop_sample"]

PATH_STYLES = "data/styles.json"
PATH_STYLE_VECTOR = "data/style_vector.json"


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_ARRAY
    dir_synced = dir_dataset / DIR_NAME_SYNCED

    songs = list(dir_input.glob("*/"))
    songs = sorted(songs)

    if (not args.overwrite) and Path(PATH_STYLES).exists():
        with open(PATH_STYLES, "r") as f:
            styles = json.load(f)
    else:
        styles = extract_styles(songs, dir_synced)
        with open(PATH_STYLES, "w") as f:
            json.dump(styles, f)

    style_vector, params = extract_style_vectors(styles)
    out = {
        "style_vector": style_vector,
        "params": params
    }
    with open(PATH_STYLE_VECTOR, "w") as f:
        json.dump(out, f)


def _load_midi(path):
    midi = np.load(path)
    onset = midi["onset"]
    mpe = midi["mpe"]
    velocity = midi["velocity"]
    velocity[onset == 0] = 0
    return mpe, velocity

def _get_style(midi, bpm):
    mpe, velocity = midi
    length = len(mpe)
    n_frames_measure = 4 * int(60 / bpm * SR / HOP_LENGTH)

    velocities = velocity[velocity.astype(bool)]
    velocities = {v: int((velocities == v).sum()) for v in range(N_VELOCITY)}

    pitches = {}
    for pitch, seq in enumerate(mpe.T):
        pitches[pitch] = int(seq.sum())

    onset_ratio = []
    on_ratio = []
    for i in range(0, length, n_frames_measure):
        seg_mpe = mpe[i:i + n_frames_measure]
        seg_vel = velocity[i:i + n_frames_measure]
        if seg_vel.any():
            on_ratio.append(seg_mpe.mean())
            onset_ratio.append(seg_vel.astype(bool).mean())

    return velocities, pitches, onset_ratio, on_ratio

def extract_styles(songs, dir_synced):
    styles = {}
    for song in tqdm(songs, desc="Extracting styles"):
        orig, = list((dir_synced / song.stem).glob("*.wav"))
        y, sr = librosa.load(str(orig))
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        pianos = list((song / DIR_NAME_PIANO).glob("*.npz"))
        pianos = sorted(pianos)

        for piano in pianos:
            midi = _load_midi(piano)
            velocities, pitches, onset_ratio, on_ratio = _get_style(midi, bpm[0])
            styles[piano.stem] = {
                "velocities": velocities,
                "pitches": pitches,
                "onset_ratio": onset_ratio,
                "on_ratio": on_ratio
            }
        with open(PATH_STYLES, "w") as f:
            json.dump(styles, f, indent=2)
    return styles


def _dict_update(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[k] += v
        else:
            d1[k] = v
    return d1

def _get_params(data):
    if isinstance(data, dict):
        sum_val = 0
        sum_div = 0
        w = 0
        for k, v in data.items():
            sum_val += int(k) * v
            w += v
        mean = sum_val / w
        for k, v in data.items():
            sum_div += v * (int(k) - mean) ** 2
        std = np.sqrt(sum_div / w)
    else:
        mean = np.mean(data)
        std = np.std(data)
    return mean, std

def _normalize(data, mean, std):
    return (data - mean) / std

def _get_dist(data, n=8):
    digits = np.digitize(data, np.linspace(-2, 2, n - 1))
    dist = np.array([(digits == i).sum() for i in range(n)])
    dist = dist / dist.sum()
    return dist

def _to_list(data):
    l = []
    for k, v in data.items():
        l += [int(k)] * v
    return l

def extract_style_vectors(styles):
    all_velocities = {}
    all_pitches = {}
    all_onset_ratio = []
    all_on_ratio = []
    for style in styles.values():
        all_velocities = _dict_update(all_velocities, style["velocities"])
        all_pitches = _dict_update(all_pitches, style["pitches"])
        all_onset_ratio += style["onset_ratio"]
        all_on_ratio += style["on_ratio"]

    mean_vel, std_vel = _get_params(all_velocities)
    mean_pitch, std_pitch = _get_params(all_pitches)
    mean_onset_ratio, std_onset_ratio = _get_params(all_onset_ratio)
    mean_on_ratio, std_on_ratio = _get_params(all_on_ratio)

    params = {
        "mean_vel": mean_vel,
        "std_vel": std_vel,
        "mean_pitch": mean_pitch,
        "std_pitch": std_pitch,
        "mean_onset_ratio": mean_onset_ratio,
        "std_onset_ratio": std_onset_ratio,
        "mean_on_ratio": mean_on_ratio,
        "std_on_ratio": std_on_ratio
    }

    style_vectors = {}
    for id_piano, style in tqdm(styles.items(), desc="Extracting style vectors"):
        velocities = np.array(_to_list(style["velocities"]))
        pitches = np.array(_to_list(style["pitches"]))
        onset_ratio = np.array(style["onset_ratio"])
        on_ratio = np.array(style["on_ratio"])

        velocities = _normalize(velocities, mean_vel, std_vel)
        pitches = _normalize(pitches, mean_pitch, std_pitch)
        onset_ratio = _normalize(onset_ratio, mean_onset_ratio, std_onset_ratio)
        on_ratio = _normalize(on_ratio, mean_on_ratio, std_on_ratio)

        dist_vel = _get_dist(velocities)
        dist_pitch = _get_dist(pitches)
        dist_onset_ratio = _get_dist(onset_ratio)
        dist_on_ratio = _get_dist(on_ratio)

        style_vectors[id_piano] = np.concatenate(
            [dist_vel, dist_pitch, dist_onset_ratio, dist_on_ratio]
        ).tolist()

    return style_vectors, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)